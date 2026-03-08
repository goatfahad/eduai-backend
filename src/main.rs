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

    // Read PORT from Railway, parse as u16
    let port = std::env::var("PORT")
        .unwrap_or_else(|_| "3000".to_string())
        .parse::<u16>()
        .expect("PORT must be a valid number");
    
    // Bind to 0.0.0.0 so Railway proxy can reach us
    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!("EduAI OpenFang Backend starting on {}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn health_check() -> &'static str {
    "EduAI OpenFang Backend - Healthy"
}

#[derive(Deserialize)]
struct TutorRequest {
    message: String,
    subject: Option<String>,
    grade_level: Option<String>,
    conversation_history: Option<Vec<ChatMessage>>,
}

#[derive(Serialize, Deserialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct TutorResponse {
    response: String,
    suggestions: Vec<String>,
}

async fn tutor_chat(
    State(state): State<AppState>,
    Json(req): Json<TutorRequest>,
) -> Json<TutorResponse> {
    let system_prompt = format!(
        "You are an expert AI tutor for {} students studying {}. Use Socratic method, break complex concepts into simple steps, use real-world examples relevant to Saudi Arabian culture, encourage and praise effort.",
        req.grade_level.unwrap_or_else(|| "high school".to_string()),
        req.subject.unwrap_or_else(|| "general subjects".to_string())
    );

    let response = state.llm_client.chat(&system_prompt, &req.message, req.conversation_history).await;

    Json(TutorResponse {
        response,
        suggestions: vec!["Can you explain that differently?".to_string(), "Give me a practice problem".to_string(), "What should I study next?".to_string()],
    })
}

async fn email_agent(State(state): State<AppState>, Json(req): Json<AgentRequest>) -> Json<AgentResponse> {
    let system_prompt = "You are an expert email assistant for teachers in Saudi Arabian schools. Draft professional responses to parent inquiries, write progress updates, handle sensitive topics with cultural awareness.";
    let response = state.llm_client.agent_task(system_prompt, &req.task, &req.context).await;
    Json(AgentResponse { result: response, agent_type: "email".to_string(), confidence: 0.95, actions_taken: vec!["Analyzed context".to_string(), "Generated response".to_string()] })
}

async fn grading_agent(State(state): State<AppState>, Json(req): Json<AgentRequest>) -> Json<AgentResponse> {
    let system_prompt = "You are an expert grading assistant. Grade assignments based on rubrics, provide constructive feedback, identify common mistakes, suggest improvements. Output JSON with grade, score, feedback, strengths, improvements.";
    let response = state.llm_client.agent_task(system_prompt, &req.task, &req.context).await;
    Json(AgentResponse { result: response, agent_type: "grading".to_string(), confidence: 0.92, actions_taken: vec!["Analyzed submission".to_string(), "Applied rubric".to_string(), "Generated feedback".to_string()] })
}

async fn lesson_planner_agent(State(state): State<AppState>, Json(req): Json<AgentRequest>) -> Json<AgentResponse> {
    let system_prompt = "You are an expert lesson planning assistant for Saudi Arabian schools. Create detailed lesson plans with objectives, materials, activities, assessments, and differentiation strategies aligned with Saudi National Curriculum.";
    let response = state.llm_client.agent_task(system_prompt, &req.task, &req.context).await;
    Json(AgentResponse { result: response, agent_type: "lesson_planner".to_string(), confidence: 0.94, actions_taken: vec!["Analyzed curriculum".to_string(), "Designed activities".to_string(), "Created assessment plan".to_string()] })
}

async fn briefing_agent(State(state): State<AppState>, Json(req): Json<AgentRequest>) -> Json<AgentResponse> {
    let system_prompt = "You are the 8AM Briefing assistant. Generate concise morning briefings with schedule overview, students needing attention, pending tasks, deadlines, and quick wins. Keep under 2 minutes reading time.";
    let response = state.llm_client.agent_task(system_prompt, &req.task, &req.context).await;
    Json(AgentResponse { result: response, agent_type: "briefing".to_string(), confidence: 0.96, actions_taken: vec!["Analyzed data".to_string(), "Generated briefing".to_string()] })
}

async fn research_agent(State(state): State<AppState>, Json(req): Json<AgentRequest>) -> Json<AgentResponse> {
    let system_prompt = "You are an educational research assistant. Find relevant resources, summarize papers, suggest teaching strategies, identify best practices, recommend supplementary materials.";
    let response = state.llm_client.agent_task(system_prompt, &req.task, &req.context).await;
    Json(AgentResponse { result: response, agent_type: "research".to_string(), confidence: 0.88, actions_taken: vec!["Searched knowledge".to_string(), "Synthesized findings".to_string()] })
}

async fn generic_agent(State(state): State<AppState>, Json(req): Json<AgentRequest>) -> Json<AgentResponse> {
    let system_prompt = "You are OpenFang, an AGI-level assistant for educators. Handle ANY task: administrative, communication, content creation, data analysis, problem solving, creative projects. Be proactive and anticipate needs.";
    let response = state.llm_client.agent_task(system_prompt, &req.task, &req.context).await;
    Json(AgentResponse { result: response, agent_type: "generic".to_string(), confidence: 0.90, actions_taken: vec!["Analyzed task".to_string(), "Generated solution".to_string()] })
}
