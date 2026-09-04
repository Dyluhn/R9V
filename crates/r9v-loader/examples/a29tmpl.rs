//! A2.9 template parity probe (dev tool): renders every golden case and
//! prints diffs. The committed gate is `tests/template_parity.rs`; this
//! example exists for fast iteration on mismatches.

use r9v_loader::{ChatContext, ChatMessage, ToolCall};
use serde_json::Value;

#[path = "../tests/common/ordered_json.rs"]
mod ordered_json;
use ordered_json::OrderedValue;

fn str_field(v: &OrderedValue, key: &str) -> String {
    v.get(key).and_then(|f| f.as_str()).unwrap_or("").to_owned()
}

fn message(v: &OrderedValue) -> ChatMessage {
    let mut m = ChatMessage {
        role: str_field(v, "role"),
        content: str_field(v, "content"),
        tool_calls: Vec::new(),
        reasoning_content: None,
    };
    if let Some(calls) = v.get("tool_calls").and_then(|c| c.as_array()) {
        for call in calls {
            let function = call.get("function");
            m.tool_calls.push(ToolCall {
                name: function.map(|f| str_field(f, "name")).unwrap_or_default(),
                arguments: function
                    .map(|f| str_field(f, "arguments"))
                    .unwrap_or_default(),
                id: call.get("id").and_then(|i| i.as_str()).map(str::to_owned),
            });
        }
    }
    m
}

fn main() {
    let dir = format!("{}/tests/fixtures", env!("CARGO_MANIFEST_DIR"));
    let golden: Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{dir}/golden-templates.json")).unwrap(),
    )
    .unwrap();
    // Reconstruct contexts (must match gen_template_fixtures.py CONTEXTS),
    // preserving document key order for `.items()` parity.
    let contexts =
        OrderedValue::parse(&std::fs::read_to_string(format!("{dir}/contexts.json")).unwrap())
            .unwrap();
    let mut fails = 0;
    let mut total = 0;
    for case in golden["cases"].as_array().unwrap() {
        let tmpl = case["template"].as_str().unwrap();
        let ctx_name = case["context"].as_str().unwrap();
        total += 1;
        let source = std::fs::read_to_string(format!("{dir}/template-{tmpl}.jinja")).unwrap();
        let ctx = contexts.get(ctx_name).expect("context must exist");
        let mut chat = ChatContext {
            messages: Vec::new(),
            add_generation_prompt: false,
            bos_token: None,
            eos_token: None,
            enable_thinking: None,
            tools: None,
            tool_choice: None,
            extra: Default::default(),
        };
        build_chat(ctx, &mut chat);
        match r9v_loader::render_chat_template(&source, &chat) {
            Ok(got) => {
                if case["status"] == "ok" && got == case["output"].as_str().unwrap() {
                    // match
                } else {
                    fails += 1;
                    println!(
                        "DIFF {tmpl}/{ctx_name}:\n--- golden ---\n{:?}\n--- got ---\n{:?}\n",
                        case["output"].as_str().unwrap_or("<err>"),
                        got
                    );
                }
            }
            Err(e) => {
                if case["status"] == "error" {
                    // both fail: parity on failure
                } else {
                    fails += 1;
                    println!(
                        "ERROR {tmpl}/{ctx_name}: {e}\ngolden={:?}",
                        case["output"].as_str().unwrap_or("<err>")
                    );
                }
            }
        }
    }
    println!("{}/{} match", total - fails, total);
}

fn ctx_messages(ctx: &OrderedValue) -> Vec<ChatMessage> {
    ctx.get("messages")
        .and_then(|m| m.as_array())
        .unwrap()
        .iter()
        .map(message)
        .collect()
}

fn build_chat(ctx: &OrderedValue, chat: &mut ChatContext) {
    chat.messages = ctx_messages(ctx);
    chat.add_generation_prompt = ctx
        .get("add_generation_prompt")
        .and_then(|f| f.as_bool())
        .unwrap();
    chat.bos_token = ctx
        .get("bos_token")
        .and_then(|f| f.as_str())
        .map(str::to_owned);
    chat.eos_token = ctx
        .get("eos_token")
        .and_then(|f| f.as_str())
        .map(str::to_owned);
    if let Some(tools) = ctx.get("tools") {
        chat.tools = Some(ordered_json::to_template_value(tools));
    }
}
