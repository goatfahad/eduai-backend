use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;

pub struct LLMClient {
      client: Client,
      groq_api_key: String,
      together_api_key: String,
}

#[derive(Serialize)]
struct GroqRequest {
      model: String,
      messages: Vec<Message>,
      temperature: f32,
      max_tokens: u32,
}

#[derive(Serialize, Deserialize, Clone)]
struct Message { role: String, content: String }

#[derive(Deserialize)]
struct GroqResponse { choices: Vec<Choice> }

#[derive(Deserialize)]
struct Choice { message: Message }

impl LLMClient {
      pub fn new() -> Self {
                Self {
                              client: Client::new(),
                              groq_api_key: env::var("GROQ_API_KEY").unwrap_or_default(),
                              together_api_key: env::var("TOGETHER_API_KEY").unwrap_or_default(),
                }
      }

    pub async fn chat(&self, system_prompt: &str, user_message: &str, history: Option<Vec<super::ChatMessage>>) -> String {
              let mut messages = vec![Message { role: "system".to_string(), content: system_prompt.to_string() }];
              if let Some(hist) = history {
                            for msg in hist { messages.push(Message { role: msg.role, content: msg.content }); }
              }
              messages.push(Message { role: "user".to_string(), content: user_message.to_string() });
              self.call_groq(messages).await
    }

    pub async fn agent_task(&self, system_prompt: &str, task: &str, context: &Option<String>) -> String {
              let full_prompt = match context {
                            Some(ctx) => format!("Task: {}\nContext: {}", task, ctx),
                            None => format!("Task: {}", task),
              };
              let messages = vec![
                            Message { role: "system".to_string(), content: system_prompt.to_string() },
                            Message { role: "user".to_string(), content: full_prompt },
                        ];
              let result = self.call_groq(messages.clone()).await;
              if result.is_empty() { self.call_together(messages).await } else { result }
    }

    async fn call_groq(&self, messages: Vec<Message>) -> String {
              let request = GroqRequest { model: "llama-3.3-70b-versatile".to_string(), messages, temperature: 0.7, max_tokens: 4096 };
              match self.client.post("https://api.groq.com/openai/v1/chat/completions")
                  .header("Authorization", format!("Bearer {}", self.groq_api_key))
                  .header("Content-Type", "application/json")
                  .json(&request).send().await {
                            Ok(response) => {
                                              if let Ok(data) = response.json::<GroqResponse>().await {
                                                                    if let Some(choice) = data.choices.first() { return choice.message.content.clone(); }
                                              }
                                              String::new()
                            }
                            Err(_) => String::new(),
              }
    }

    async fn call_together(&self, messages: Vec<Message>) -> String {
              let request = serde_json::json!({ "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "messages": messages, "temperature": 0.7, "max_tokens": 4096 });
              match self.client.post("https://api.together.xyz/v1/chat/completions")
                  .header("Authorization", format!("Bearer {}", self.together_api_key))
                  .header("Content-Type", "application/json")
                  .json(&request).send().await {
                            Ok(response) => {
                                              if let Ok(data) = response.json::<GroqResponse>().await {
                                                                    if let Some(choice) = data.choices.first() { return choice.message.content.clone(); }
                                              }
                                              String::new()
                            }
                            Err(_) => String::new(),
              }
    }
}
