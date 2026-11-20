//! Each model's own chat template on multi-turn histories, which a single user message never renders.

use clap::ValueEnum;
use either::Either;
use indexmap::IndexMap;
use mistralrs::{TextMessageRole, TextMessages};
use mistralrs_core::tools::{process_model_specific_message, CalledFunctionParameters};
use mistralrs_core::{apply_chat_template_to, ChatTemplateValue, MessageContent, Tool};
use mistralrs_server_core::chat_completion::template_message;
use oscar_models::rows::ModelPreset;
use serde_json::{json, Value};

use crate::helpers::fixture;

type Message = IndexMap<String, MessageContent>;

fn msg(role: &str, content: &str, reasoning: Option<&str>) -> Message {
    let fields = [("role", Some(role)), ("content", Some(content)), ("reasoning_content", reasoning)];
    fields.into_iter().filter_map(|(k, v)| Some((k.to_string(), Either::Left(v?.to_string())))).collect()
}

/// `(role, content)` turns without reasoning.
fn msgs(turns: &[(&str, &str)]) -> Vec<Message> {
    turns.iter().map(|&(role, content)| msg(role, content, None)).collect()
}

/// An assistant turn of calls alone, its content as the client sends it, converted as serve converts it.
fn call_turn(content: Value, calls: &[(&str, Value, &str)]) -> Message {
    let call = |(name, args, id): &(&str, Value, &str)| json!({"id": id, "type": "function", "function": {"name": name, "arguments": args.to_string()}});
    let calls: Vec<Value> = calls.iter().map(call).collect();
    template_message(
        &serde_json::from_value(json!({"role": "assistant", "content": content, "tool_calls": calls})).unwrap(),
    )
    .unwrap()
}

fn tool_result(id: &str, content: &str) -> Message {
    let mut m = msg("tool", content, None);
    m.insert("tool_call_id".to_string(), Either::Left(id.to_string()));
    m
}

fn tool(function: Value) -> Tool {
    serde_json::from_value(json!({"type": "function", "function": function})).unwrap()
}

/// `template` as serve renders it, `thinking` the request's `enable_thinking`.
fn render_template(
    template: &str, messages: &[Message], add_generation_prompt: bool, thinking: Option<bool>, tools: Vec<Tool>,
) -> anyhow::Result<String> {
    let template = ChatTemplateValue(Either::Left(template.to_string()));
    apply_chat_template_to(messages.to_vec(), add_generation_prompt, thinking, None, &template, None, None, None, tools)
}

fn render(preset: ModelPreset, messages: &[Message], add_generation_prompt: bool, tools: Vec<Tool>) -> String {
    render_template(preset.def().chat_template, messages, add_generation_prompt, None, tools)
        .unwrap_or_else(|e| panic!("{preset:?}: {e}"))
}

/// Two exchanges with reasoning returned, and a third user turn making them history.
const CONVERSATION: [(&str, &str, Option<&str>); 5] = [
    ("user", "which file holds the score?", None),
    ("assistant", "In `game.py`.", Some("They mean the high score table.")),
    ("user", "add a reset command", None),
    ("assistant", "Added.", Some("Resetting means truncating the table, not deleting it.")),
    ("user", "does it persist?", None),
];

fn conversation() -> Vec<Message> {
    CONVERSATION.iter().map(|&(role, content, reasoning)| msg(role, content, reasoning)).collect()
}

/// Historical thinking survives a new user turn; a turn of calls alone renders alike with null or empty content;
/// the renderer keeps a loop's `set` past its iteration, so a template resets it or a turn inherits thinking; the
/// context cache resumes only where the earlier prompt is a byte prefix; in-process history renders as the wire one.
#[test]
fn every_models_template_renders_a_history_the_same_however_it_is_built() {
    let wire = conversation();
    let mut history = TextMessages::new();
    for (role, text, reasoning) in CONVERSATION {
        history = match role {
            "user" => history.add_message(TextMessageRole::User, text),
            _ => history.add_message_with_reasoning(TextMessageRole::Assistant, text, reasoning.unwrap_or_default()),
        };
    }
    let in_process: Vec<Message> = history.into();
    let call_only = |content| [msg("user", "x", None), call_turn(content, &[("read", json!({"path": "a.rs"}), "c1")])];
    let thinking = |last| {
        [msg("user", "think first", None), msg("assistant", "thought about it", Some("some thinking"))]
            .into_iter()
            .chain([msg("user", "hi", None), msg("assistant", "hello", last)])
            .collect::<Vec<_>>()
    };
    for &preset in ModelPreset::value_variants() {
        let rendered = |turns: &[Message], prompt| render(preset, turns, prompt, vec![]);
        // Qwen3-Coder-Next has no reasoning channel: a client's `reasoning_content` for it is foreign text.
        if !matches!(preset, ModelPreset::Qwen3CoderNext) {
            let out = rendered(&wire, true);
            assert!(out.contains("They mean the high score table."), "{preset:?}: {out}");
            assert!(out.contains("Resetting means truncating the table, not deleting it."), "{preset:?}: {out}");
        }
        assert_eq!(rendered(&call_only(Value::Null), false), rendered(&call_only(json!("")), false), "{preset:?}");
        assert_eq!(rendered(&thinking(Some("")), false), rendered(&thinking(None), false), "{preset:?}");
        let (earlier, later) = (rendered(&wire[..2], false), rendered(&wire, false));
        assert!(later.starts_with(&earlier), "{preset:?}:\n{earlier}\n---\n{later}");
        assert_eq!(rendered(&in_process, true), rendered(&wire, true), "{preset:?}");
    }
}

/// `enable_thinking=false` is what `+nothink` and `--no-think` render: GLM-5.3-Flash closes the think block. What the
/// parse-and-render round trip changes is re-encoded next request, and a dropped indent reaches the edit.
#[test]
fn tool_turns_render_the_published_form_and_a_call_renders_back_as_the_model_wrote_it() {
    let turns = [
        msg("user", "what does game.py hold?", None),
        call_turn(Value::Null, &[("read_file", json!({"path": "game.py", "limit": 40}), "c1")]),
        msg("tool", "def main(): ...", None),
        msg("tool", "score_table = {}", None),
        msg("user", "thanks", None),
    ];
    let qwen = ["<tool_call>\n<function=read_file>", "<parameter=path>"];
    let glm = [
        "<tool_call>read_file<arg_key>path</arg_key><arg_value>game.py</arg_value>",
        "<arg_key>limit</arg_key><arg_value>40</arg_value>",
        "<tool_response>def main(): ...</tool_response>",
    ];
    let qwen_counts = [("<tool_response>", 2), ("<|im_start|>user", 3)];
    for (preset, has, counts, tail) in [
        // Two responses share a user turn through `loop.previtem`/`loop.nextitem`, the one minijinja construct to pin.
        (ModelPreset::Qwen3CoderNext, &qwen[..], &qwen_counts[..], "<|im_start|>assistant\n"),
        (ModelPreset::Qwen38FlashNext, &qwen, &qwen_counts, "<|im_start|>assistant\n<think>\n"),
        // The form `parse_glm_tool_call` reads, strings raw and the rest JSON; two results share an observation block.
        (ModelPreset::Glm53Flash, &glm, &[("<|observation|>", 1), ("<tool_response>", 2)], "<|assistant|><think>"),
    ] {
        let out = render(preset, &turns, true, vec![]);
        has.iter().for_each(|s| assert!(out.contains(s), "{preset:?}: no {s:?} in {out}"));
        counts.iter().for_each(|&(s, n)| assert_eq!(out.matches(s).count(), n, "{preset:?}: {s:?} in {out}"));
        assert!(out.ends_with(tail), "{preset:?}: {out}");
    }
    let template = ModelPreset::Glm53Flash.def().chat_template;
    let turns = [msg("user", "name this conversation", None)];
    let render_thinking = |thinking| render_template(template, &turns, true, thinking, vec![]).unwrap();
    let off = render_thinking(Some(false));
    assert!(off.ends_with("<|assistant|><think></think>") && !off.contains("Reasoning Effort:"), "{off}");
    let on = render_thinking(None);
    assert!(on.ends_with("<|assistant|><think>") && on.contains("<|system|>Reasoning Effort: Max"), "{on}");
    assert_eq!(on, render_thinking(Some(true)));
    // The round trip.
    let qwen_xml = [
        "<tool_call>\n<function=edit>\n<parameter=filePath>\ngame.py\n</parameter>\n\
         <parameter=oldString>\n    x = 1\n    y = 2\n</parameter>\n\
         <parameter=newString>\n    x = 3\n\n</parameter>\n</function>\n</tool_call>",
        "<tool_call>\n<function=question>\n<parameter=questions>\n\
         [{\"header\": \"Edit scope\", \"options\": [{\"label\": \"Yes\", \"description\": \"Add <coins> & don't stop\"}]}]\n\
         </parameter>\n</function>\n</tool_call>",
    ];
    let glm = [
        "<tool_call>edit<arg_key>filePath</arg_key><arg_value>game.py</arg_value>\
         <arg_key>oldString</arg_key><arg_value>    x = 1\n    y = 2</arg_value>\
         <arg_key>newString</arg_key><arg_value>    x = 3\n</arg_value></tool_call>",
        "<tool_call>question<arg_key>questions</arg_key><arg_value>\
         [{\"header\": \"Edit scope\", \"options\": [{\"label\": \"Yes\", \"description\": \"Add <coins> & don't stop\"}]}]\
         </arg_value></tool_call>",
    ];
    for (preset, calls) in [
        (ModelPreset::Qwen397B, qwen_xml),
        (ModelPreset::Qwen35B, qwen_xml),
        (ModelPreset::Qwen38FlashNext, qwen_xml),
        (ModelPreset::Qwen3CoderNext, qwen_xml),
        (ModelPreset::Glm52, glm),
        (ModelPreset::Glm53Flash, glm),
    ] {
        for written in calls {
            let parsed: CalledFunctionParameters =
                serde_json::from_str(&process_model_specific_message(written).unwrap())
                    .unwrap_or_else(|e| panic!("{preset:?}: the call does not parse ({e})"));
            if parsed.name == "edit" {
                assert_eq!(parsed.parameters["oldString"], "    x = 1\n    y = 2", "{preset:?}");
                assert_eq!(parsed.parameters["newString"], "    x = 3\n", "{preset:?}");
            } else {
                assert!(parsed.parameters["questions"].is_array(), "{preset:?}: the argument is JSON");
            }
            let turns = [
                msg("user", "fix it", None),
                call_turn(Value::Null, &[(&parsed.name, parsed.parameters, "c1")]),
                msg("tool", "done", None),
                msg("user", "thanks", None),
            ];
            let out = render(preset, &turns, true, vec![]);
            assert!(out.contains(written), "{preset:?}: the history does not carry the call as written:\n{out}");
        }
    }
}

/// The expected strings are transformers' own renders: templates are written against its `tojson`.
#[test]
fn tojson_writes_what_transformers_writes() {
    let read = tool(json!({"name": "read", "description": "Read <path> & say it's done — café", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]
    }}));
    for (form, want) in [
        ("tojson", "{\"type\": \"function\", \"function\": {\"name\": \"read\", \"description\": \"Read <path> & say it's done — café\", \"parameters\": {\"type\": \"object\", \"properties\": {\"path\": {\"type\": \"string\"}, \"limit\": {\"type\": \"integer\"}}, \"required\": [\"path\"]}}}"),
        ("tojson(indent=2)", "{\n  \"type\": \"function\",\n  \"function\": {\n    \"name\": \"read\",\n    \"description\": \"Read <path> & say it's done — café\",\n    \"parameters\": {\n      \"type\": \"object\",\n      \"properties\": {\n        \"path\": {\n          \"type\": \"string\"\n        },\n        \"limit\": {\n          \"type\": \"integer\"\n        }\n      },\n      \"required\": [\n        \"path\"\n      ]\n    }\n  }\n}"),
        ("tojson(separators=(',', ':'))", "{\"type\":\"function\",\"function\":{\"name\":\"read\",\"description\":\"Read <path> & say it's done — café\",\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"limit\":{\"type\":\"integer\"}},\"required\":[\"path\"]}}}"),
        ("tojson(sort_keys=true)", "{\"function\": {\"description\": \"Read <path> & say it's done — café\", \"name\": \"read\", \"parameters\": {\"properties\": {\"limit\": {\"type\": \"integer\"}, \"path\": {\"type\": \"string\"}}, \"required\": [\"path\"], \"type\": \"object\"}}, \"type\": \"function\"}"),
        ("tojson(ensure_ascii=true)", "{\"type\": \"function\", \"function\": {\"name\": \"read\", \"description\": \"Read <path> & say it's done \\u2014 caf\\u00e9\", \"parameters\": {\"type\": \"object\", \"properties\": {\"path\": {\"type\": \"string\"}, \"limit\": {\"type\": \"integer\"}}, \"required\": [\"path\"]}}}"),
    ] {
        let template = format!("{{{{ tools[0] | {form} }}}}");
        let out = render_template(&template, &[msg("user", "hi", None)], false, None, vec![read.clone()]).unwrap();
        assert_eq!(out, want, "{form}");
    }
}

/// `,cpp` tokenises as `,c` `pp` though `cpp` is a token; `<|end|>` stands for a template's turn marker.
const SPLIT_TOKENIZER: &str = r#"{"version": "1.0", "truncation": null, "padding": null, "added_tokens": [{"id": 10, "content": "<|end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}], "normalizer": null, "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}, "post_processor": null, "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}, "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null, "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "vocab": {"a": 0, "b": 1, ",": 2, "c": 3, "p": 4, ",c": 5, "pp": 6, "cpp": 7, "x": 8, "y": 9}, "merges": [", c", "p p", "c pp"]}}"#;

/// A model samples tokens the tokenizer would not write, so a re-tokenised history would part from the cached turn.
#[test]
fn a_history_that_repeats_a_turn_keeps_the_models_own_tokens() {
    use std::str::FromStr;
    let tok = tokenizers::Tokenizer::from_str(SPLIT_TOKENIZER).unwrap();
    let ids = |s: &str| tok.encode_fast(s, false).unwrap().get_ids().to_vec();
    let note = |toks: Vec<u32>| {
        let mut turns = mistralrs_core::prefix_cacher::Turns::default();
        turns.note(tok.decode(&toks, false).unwrap(), toks, 2);
        turns
    };
    // A prompt "ab" and a reply the model sampled as `,` `cpp`.
    let turns = note(vec![0, 1, 2, 7]);
    let next = "ab,cpp<|end|>xy";
    assert_eq!(ids(next), [0, 1, 5, 6, 10, 8, 9], "the tokenizer writes `,cpp` as `,c` `pp`");
    assert_eq!(turns.own_tokens(next, ids(next), &tok), [0, 1, 2, 7, 10, 8, 9]);
    let parted = "ab,cp<|end|>xy";
    assert_eq!(turns.own_tokens(parted, ids(parted), &tok), ids(parted), "a history that parts from the turn's text");
    // A turn ending inside a token of the whole render: `,c` `pp` spans the cut.
    let turns = note(vec![0, 1, 2, 3, 4]);
    assert_eq!(turns.own_tokens(next, ids(next), &tok), ids(next), "a cut the whole tokenization does not make");
}

/// The goldens are the vendor's template rendered by Jinja2 on these inputs; oscar renders through the mistral.rs fork.
#[test]
fn k3_renders_as_the_vendors_jinja2_does() {
    let weather = || {
        tool(json!({"name": "get_weather", "description": "Weather by city.", "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "units": {"type": "string", "enum": ["c", "f"]}},
            "required": ["city"]
        }}))
    };
    let every_type = json!({
        "s": "plain", "n": 42, "f": 2.0, "b": true, "z": null,
        "arr": [1, "two", 3.5], "obj": {"k": "v", "d": {"x": 1}},
        "we\"ird&": "esc"
    });
    let call_every_type = call_turn(Value::Null, &[("t", every_type, "c1")]);
    let two_calls = call_turn(Value::Null, &[("a", json!({"p": 1}), "id-a"), ("b", json!({"q": 2}), "id-b")]);
    let paris_then_london = call_turn(
        Value::Null,
        &[("get_weather", json!({"city": "Paris"}), "p1"), ("get_weather", json!({"city": "London"}), "p2")],
    );
    let reordered = [msgs(&[("user", "x")]), vec![two_calls, tool_result("id-b", "rb"), tool_result("id-a", "ra")]];
    let full = [
        msgs(&[("system", "Be terse."), ("user", "weather in Paris then London")]),
        vec![paris_then_london, tool_result("p1", "18C"), tool_result("p2", "15C")],
        msgs(&[("assistant", "Paris 18C, London 15C."), ("user", "thanks")]),
    ];
    for (golden, turns, add_generation_prompt, tools) in [
        ("declare_two_tools", msgs(&[("user", "hi")]), true, vec![weather(), tool(json!({"name": "noop"}))]),
        ("call_every_argument_type", vec![msg("user", "x", None), call_every_type], false, vec![]),
        ("results_reordered_by_id", reordered.concat(), true, vec![]),
        ("full_conversation", full.concat(), true, vec![weather()]),
    ] {
        let want = String::from_utf8(fixture(&format!("k3_render_goldens/{golden}.golden"))).unwrap();
        assert_eq!(render(ModelPreset::KimiK3, &turns, add_generation_prompt, tools), want, "{golden}");
    }
}
