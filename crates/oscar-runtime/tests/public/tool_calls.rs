//! Tool-call parsing: each family's form, and the fallback surfacing calls the tag parser files as reasoning.

use std::sync::Arc;

use mistralrs_core::reasoning_parsers::TagReasoningContext;
use mistralrs_core::tools::{
    parse_tools_from_reasoning_output, process_model_specific_message, CalledFunctionParameters, ToolCallingMatcher,
};
use mistralrs_core::{ToolCallResponse, ToolChoice};
use serde_json::{json, Value};

/// The fallback as sampling.rs drives it, with an auto matcher: what content keeps, and the calls.
fn parse(content: Option<&str>, reasoning: Option<&str>, raw: Option<&str>) -> (Option<String>, Vec<ToolCallResponse>) {
    let matcher = Arc::new(ToolCallingMatcher::new(ToolChoice::Auto).unwrap());
    parse_tools_from_reasoning_output(content, reasoning, raw, Some(matcher)).unwrap()
}

/// Content keeps precedence and its own call consumes it; the think block's and then the raw completion's leave it.
/// Why the fallback is needed at all: the tag parser routes think-block tool text into reasoning. Each form parses
/// to its calls and a cut one to none; K3's terminal calls reach the client from the raw completion.
#[test]
fn a_call_comes_from_content_then_the_think_block_then_the_raw_completion_in_every_form() {
    let think = "Let me check the weather first.\n<tool_call>\n\
                 {\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
    let from_content = "<tool_call>\n{\"name\": \"from_content\", \"arguments\": {}}\n</tool_call>";
    let from_think = "<tool_call>\n{\"name\": \"from_think\", \"arguments\": {}}\n</tool_call>";
    let from_raw = "ignored<|open|>tools<|sep|><|open|>call tool=\"from_raw\" index=\"1\"<|sep|>\
                    <|close|>call<|sep|><|close|>tools<|sep|>";
    // Kimi K3's call block, cut mid-argument.
    let k3_cut = "t<|close|>think<|sep|><|open|>response<|sep|>a<|close|>response<|sep|>\
                  <|open|>tools<|sep|><|open|>call tool=\"get_weather\" index=\"1\"<|sep|>\
                  <|open|>argument key=\"ci";
    // OpenAI's form, the arguments a JSON string: a call, and while cut a possible one, so none of it streams as text.
    let openai = r#"[{"name":"write","arguments":"{\"path\":\"a.rs\",\"content\":\"fn main() {}\\n\"}"}]"#;
    let cut = ToolCallingMatcher::new(ToolChoice::Auto).unwrap().prefix_could_be_tool(&openai[..48]).unwrap();
    assert_eq!(cut, (true, false), "{}", &openai[..48]);
    for (content, reasoning, raw, kept, called) in [
        (Some(""), Some(think), None, Some(""), Some("get_weather")),
        (Some(from_content), Some(from_think), None, None, Some("from_content")),
        (Some("plain answer"), Some("plain thinking"), None, Some("plain answer"), None),
        (Some(from_content), None, Some(from_raw), None, Some("from_content")),
        (Some("a"), Some("t"), Some(k3_cut), Some("a"), None),
        (Some(openai), None, None, None, Some("write")),
    ] {
        let (got, calls) = parse(content, reasoning, raw);
        let names: Vec<&str> = calls.iter().map(|c| c.function.name.as_str()).collect();
        assert_eq!(names, Vec::from_iter(called), "{content:?} {reasoning:?} {raw:?}");
        assert_eq!(got.as_deref(), kept, "{content:?} {reasoning:?} {raw:?}");
    }
    let mut p = TagReasoningContext::new_think_tags();
    p.process_text("<think>I should call it: <tool_call>{\"name\": \"t\", \"arguments\": {}}</tool_call></think>done");
    p.finalize();
    assert_eq!(p.content().as_deref(), Some("done"));
    assert!(p.reasoning_content().unwrap().contains("<tool_call>"), "tool text lives in reasoning");
    // Qwen's XML and JSON forms and GLM's share the outer tag and part on their inner shape: GLM's name check keeps
    // the JSON form from naming a function after its JSON; values stay typed. K3 places its calls after the response
    // channel, where the tag parser files nothing.
    let calls = |msg: &str| serde_json::from_str::<Value>(&process_model_specific_message(msg).unwrap()).unwrap();
    for msg in [
        "\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
        "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>",
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
    ] {
        assert_eq!(calls(msg), json!({"name": "get_weather", "arguments": {"city": "Paris"}}), "{msg}");
    }
    let two =
        json!([{"name": "set_temp", "arguments": {"value": 42}}, {"name": "toggle", "arguments": {"enabled": true}}]);
    for msg in [
        "<tool_call>\n<function=set_temp>\n<parameter=value>\n42\n</parameter>\n</function>\n</tool_call>\n\
         <tool_call>\n<function=toggle>\n<parameter=enabled>\ntrue\n</parameter>\n</function>\n</tool_call>",
        "<tool_call>set_temp<arg_key>value</arg_key><arg_value>42</arg_value></tool_call>\n\
         <tool_call>toggle<arg_key>enabled</arg_key><arg_value>true</arg_value></tool_call>",
    ] {
        assert_eq!(calls(msg), two, "{msg}");
    }
    // No arguments is an empty object, not a missing field.
    assert_eq!(calls("<tool_call>list_files</tool_call>"), json!({"name": "list_files", "arguments": {}}));
    // A cut value, a missing closing tag, a dangling key; and prose, the control a call-fabricating parser fails.
    let prose = "The weather in Paris is mild today, around 18 degrees.";
    for msg in [
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Par",
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</tool_call>",
        "<tool_call>get_weather<arg_key>ci",
        prose,
    ] {
        let out = process_model_specific_message(msg).unwrap();
        assert!(serde_json::from_str::<CalledFunctionParameters>(&out).is_err(), "{msg} made a call: {out}");
    }
    assert_eq!(process_model_specific_message(prose).unwrap(), prose, "prose passes through untouched");
    let raw = "Two lookups needed.<|close|>think<|sep|><|open|>response<|sep|>\
               Checking.<|close|>response<|sep|><|open|>tools<|sep|>\
               <|open|>call tool=\"get_weather\" index=\"1\"<|sep|>\
               <|open|>argument key=\"city\" type=\"string\"<|sep|>Paris<|close|>argument<|sep|>\
               <|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|>";
    // The channel split sampling.rs hands over, from the real parser.
    let mut ctx = TagReasoningContext::new_kimi_channel(true);
    raw.as_bytes().chunks(7).for_each(|c| ctx.process_text(std::str::from_utf8(c).unwrap_or("")));
    ctx.finalize();
    let (content, reasoning) = (ctx.content(), ctx.reasoning_content());
    assert_eq!((content.as_deref(), reasoning.as_deref()), (Some("Checking."), Some("Two lookups needed.")));
    let (kept, calls) = parse(content.as_deref(), reasoning.as_deref(), Some(raw));
    let [call] = &calls[..] else { panic!("one call, got {}", calls.len()) };
    let args: Value = serde_json::from_str(&call.function.arguments).unwrap();
    assert_eq!((call.function.name.as_str(), &args["city"]), ("get_weather", &json!("Paris")));
    assert_eq!(kept.as_deref(), Some("Checking."), "the answer channel survives");
}
