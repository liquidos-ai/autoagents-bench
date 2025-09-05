#![allow(dead_code, unused_variables, unused_imports, unreachable_code)]
use autoagents::core::agent::prebuilt::executor::BasicAgent;
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{
    AgentBuilder, AgentDeriveT, AgentExecutor, AgentOutputT, BaseAgent, DirectAgent,
};
use autoagents::core::error::Error;
use autoagents::core::tool::{ToolCallError, ToolInputT, ToolRuntime, ToolT};
use autoagents::llm::LLMProvider;
use autoagents::llm::backends::openai::OpenAI;
use autoagents::llm::builder::LLMBuilder;
use autoagents_derive::{AgentOutput, ToolInput, agent, tool};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::openai;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use std::time::Instant;
use tokio::task;

#[agent(name = "simple_agent", description = "You are a helpful assistant.")]
#[derive(Default, Clone)]
pub struct SimpleAgent {}

#[derive(thiserror::Error, Debug)]
pub enum AgentError {
    #[error("LLM error")]
    LLMError,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let num_cucurrent = 50;

    let openai = openai::Client::from_env();
    // Create simple agent
    let agent = Arc::new(
        openai
            .agent("gpt-4")
            .preamble("You are a helpful assistant.")
            .temperature(0.7)
            .build(),
    );

    let start = Instant::now();
    let mut handles = vec![];

    // Spawn 10 concurrent tasks
    for i in 0..num_cucurrent {
        let model_clone = Arc::clone(&agent);
        let handle = task::spawn(async move {
            let prompt = format!("Generate a random fact about the number {i}");
            model_clone.prompt(&prompt).await
        });
        handles.push(handle);
    }

    // Collect results
    for handle in handles {
        let result = handle.await??;
        println!("Result: {result}");
    }

    println!("Time elapsed: {:?}", start.elapsed());

    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or("".into());
    let llm = LLMBuilder::<OpenAI>::new()
        .api_key(api_key)
        .model("gpt-4")
        .temperature(0.7)
        .build()
        .expect("Failed to build LLM");

    let agent = Arc::new(
        AgentBuilder::<_, DirectAgent>::new(BasicAgent::new(SimpleAgent {}))
            .llm(llm)
            .build()?,
    );

    let start = Instant::now();
    let mut handles = vec![];

    // Spawn 10 concurrent tasks
    for i in 0..num_cucurrent {
        let agent_clone = agent.clone();
        let handle = task::spawn(async move {
            let prompt = format!("Give a random fact about the number {i}");
            agent_clone.run(Task::new(&prompt)).await
        });
        handles.push(handle);
    }

    // Collect results
    for handle in handles {
        let result = handle.await??;
        println!("Result: {:?}", result);
    }

    println!("Time elapsed: {:?}", start.elapsed());
    Ok(())
}
