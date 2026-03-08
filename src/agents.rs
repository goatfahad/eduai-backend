use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct AgentRequest {
      pub task: String,
      pub context: Option<String>,
      pub agent_type: Option<String>,
      pub syllabus: Option<String>,
      pub rubric: Option<String>,
      pub student_data: Option<String>,
      pub email_thread: Option<String>,
}

#[derive(Serialize)]
pub struct AgentResponse {
      pub result: String,
      pub agent_type: String,
      pub confidence: f32,
      pub actions_taken: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentType {
      Email,
      Grading,
      LessonPlanner,
      Briefing,
      Research,
      Generic,
}

impl AgentType {
      pub fn from_str(s: &str) -> Self {
                match s.to_lowercase().as_str() {
                              "email" => AgentType::Email,
                              "grading" | "grade" => AgentType::Grading,
                              "lesson" | "lesson_planner" => AgentType::LessonPlanner,
                              "briefing" | "8am" => AgentType::Briefing,
                              "research" => AgentType::Research,
                              _ => AgentType::Generic,
                }
      }
}
