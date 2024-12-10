use axum::http::StatusCode;
use axum::response::Response;
// Custom error wrapper that holds anyhow::Error
pub struct AppError(pub anyhow::Error);

// Convert anyhow::Error into AppError
impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
}

// Implement IntoResponse for AppError to convert errors into HTTP responses
impl axum::response::IntoResponse for AppError {
    fn into_response(self) -> Response {
        // Log the error with its full chain of causes
        tracing::error!("Application error: {:#}", self.0);

        // You can match on specific error types and return different status codes
        let status = if self.0.downcast_ref::<std::io::Error>().is_some() {
            StatusCode::INTERNAL_SERVER_ERROR
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        };

        // Return the error response
        (status, format!("Something went wrong: {}", self.0)).into_response()
    }
}

impl From<AppError> for std::io::Error {
    fn from(err: AppError) -> Self {
        std::io::Error::new(std::io::ErrorKind::Other, err.0.to_string())
    }
}
