// SPDX-License-Identifier: Apache-2.0
//! The spliced chat-completions call ([`Client::chat_spliced`]): the body is [`ChatBody::body`], sent as is, and the
//! answer is kept as received ([`ChatAnswer::raw`]).

use crate::http::client::{Client, check_redirect, read};
use crate::http::error::{Error, ErrorKind};
use crate::http::native::from_native;
use crate::wire::openai_chat::{ChatAnswer, ChatBody, TokenField, error_envelope, parse};
use crate::wire::provider::{self, HttpRequest, ProviderKind};

impl Client {
    /// One non-streaming chat-completions call with a spliced body. A refusal is an answer, not an error (its view's
    /// `stop_reason` is `refusal`).
    ///
    /// # Errors
    /// The target is Anthropic (its turns go through [`Client::messages`]); any transport or provider failure.
    pub async fn chat_spliced(&self, request: &ChatBody) -> Result<ChatAnswer, Error> {
        self.chat_spliced_inner(request)
            .await
            .map_err(|e| self.scrub(e))
    }

    async fn chat_spliced_inner(&self, request: &ChatBody) -> Result<ChatAnswer, Error> {
        if !self.speaks_openai_chat() {
            return Err(Error::new(
                ErrorKind::InvalidRequest,
                "a spliced chat call needs an OpenAI-compatible target or a gateway",
            ));
        }
        let target = self.target.translate("");
        let mut http: HttpRequest = provider::openai_endpoint(&target);
        http.body = request.body();
        let resp = self.send(http, None, Some(self.timeout), false).await?;
        let request_id = resp
            .headers()
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        let (status, retry_after, body) = read(resp, self.max_response_bytes).await?;
        check_redirect(status)?;
        if status >= 300 {
            return Err(Error::from_translate(
                provider::provider_error(status, &body),
                retry_after,
            ));
        }
        if let Some(e) = error_envelope(&body) {
            return Err(Error::from_translate(e, retry_after));
        }
        let mut answer = parse(body).map_err(from_native)?;
        answer.request_id = request_id;
        Ok(answer)
    }

    /// Whether [`Client::chat_spliced`] can reach this target: an OpenAI-compatible API or a gateway.
    pub fn speaks_openai_chat(&self) -> bool {
        self.target.kind() == ProviderKind::OpenAi
    }

    /// The name this target gives the output limit: `max_completion_tokens` for OpenAI itself, `max_tokens` for
    /// every other OpenAI-compatible server and the gateway.
    pub fn chat_token_field(&self) -> TokenField {
        let target = self.target.translate("");
        if provider::uses_completion_tokens(&target.base_url) {
            TokenField::MaxCompletionTokens
        } else {
            TokenField::MaxTokens
        }
    }
}
