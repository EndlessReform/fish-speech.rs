use super::error::AppError;
use crate::state::AppState;
use axum::{extract::State, Json};
use std::sync::Arc;

pub async fn get_supported_voices(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<String>>, AppError> {
    let voice_map = state.lm.voices.lock().await;

    let voices: Vec<String> = voice_map.keys().map(|k| k.to_owned()).collect();

    Ok(Json(voices))
}
