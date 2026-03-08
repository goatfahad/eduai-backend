use axum::{
      extract::{Json, State},
      http::{HeaderValue, Method},
      routing::{get, post},
      Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tracing_subscriber;

mod agents;
mod llm;

use agents::{AgentRequest, AgentResponse};
use llm::LLMClient;

#[derive(Clone)]
pub struct AppState {
      pub llm_client: Arc<LLMClient>,
}

#[tokio::main]
async fn main() {
      tracing_subscriber::fmt::init();
      dotenv::dotenv().ok();
      let llm_client = Arc::new(LLMClient::new());
      let state = AppState { llm_client };
      let cors = CorsLayer::new()
                .allow_origin("*".parse::<HeaderValue>().unwrap())
                .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
                .allow_headers(tower_http::cors::Any);
      let app = Router::new()
                .route("/health", get(health_check))
                .route("/api/tutor", post(tutor_chat))
                .route("/api/agent/email", post(email_agent))
                .route("/api/agent/grade", post(grading_agent))
                .route("/api/agent/lesson", post(lesson_planner_agent))
                .route("/api/agent/briefing", post(briefing_agent))
                .route("/api/agent/research", post(research_agent))
                .route("/api/agent", post(generic_agent))
                .layer(cors)
                .with_state(state);
      let port = std::env::var("PORT").unwrap_or_else(|_| "3000".to_string());
      let addr = format!("0.0.0.0:{}", port);
      tracing::info!("EduAI OpenFang Backend starting on {}", addr);
      let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
      axum::serve(listener, app).await.unwrap();
}

async fn health_check() -> &'static str { "EduAI OpenFang Backend - Healthy" }

#[derive(Deserialize)]
struct TutorRequest { message: String, subject: Option<String>, grade_level: Option<String>, conversation_history: Option<Vec<ChatMessage>> }

#[derive(Serialize, Deserialize, Clone)]
struct ChatMessage { role: String, content: String }

#[derive(Serialize)]
struct TutorResponse { response: String, suggestions: Vec<String> }

async fn tutor_chat(State(state): State<AppState>, Json(req): Json<TutorRequest>) -> Json<TutorResponse> {
      let system_prompt = format!("You are an expert AI tutor for {} students studying {}. Use Socratic method.", req.grade_level.unwrap_or_else(|| "high school".to_string()), req.subject.unwrap_or_else(|| "general".to_string()));
      let response = state.llm_client.chat(&system_prompt, &req.message, req.conversation_history).await;
      Json(TutorResponse { response, suggestions: vec!["Explain differently?".to_string(), "Practice problem".to_string()] })
}

async fn email_agent(State(state): State<AppState>, Json(req): Json<AgentRequest>) -> Json<AgentResponse> {
      let response = state.llm_client.agent_task("Email assistant for teachers", &req.task, &req.context).await;
      Json(AgentResponse { result: response, agent_type: "email".to_string(), confidence: 0.95, actions_taken: vec!["Generated".to_string()] })
}

async fn grading_agent(State(state): State<AppState>, Json(req): Json<AgentRequest>) -> Json<AgentResponse> {
      let response = state.llm_client.agent_task("Grading assistant with rubrics", &req.task, &req.context).await;
      Json(AgentResponse { result: response, agent_type: "grading".to_string(), confidence: 0.92, actions_taken: vec!["Graded".to_string()] })
}

async fn lesson_planner_agent(State(state): State<AppState>, Json(req): Json<AgentRequest>) -> Json<AgentResponse> {
      let response = state.llm_client.agent_task("Lesson planner for Saudi schools", &req.task, &req.context).await;
      Json(AgentResponse { result: response, agent_type: "lesson_planner".to_string(), confidence: 0.94, actions_taken: vec!["Planned".to_string()] })
}

async fn briefing_agent(State(state): State<AppState>, Json(req): Json<AgentRequest>) -> Json<AgentResponse> {
      let response = state.llm_client.agent_task("8AM Briefing generator", &req.task, &req.context).await;
      Json(AgentResponse { result: response, agent_type: "briefing".to_string(), confidence: 0.96, actions_taken: vec!["Briefed".to_string()] })
}

async fn research_agent(State(state): State<AppState>, Json(req): Json<AgentRequest>) -> Json<AgentResponse> {
      let response = state.llm_client.agent_task("Educational research assistant", &req.task, &req.context).await;
      Json(AgentResponse { result: response, agent_type: "research".to_string(), confidence: 0.88, actions_taken: vec!["Researched".to_string()] })
}

async fn generic_agent(State(state): State<AppState>, Json(req): Json<AgentRequest>) -> Json<AgentResponse> {
      let response = state.llm_client.agent_task("OpenFang AGI for educators", &req.task, &req.context).await;
      Json(AgentResponse { result: response, agent_type: "generic".to_string(), confidence: 0.90, actions_taken: vec!["Completed".to_string()] })
}
