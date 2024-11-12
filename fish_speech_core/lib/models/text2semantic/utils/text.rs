use std::collections::HashMap;
use std::sync::OnceLock;

static SYMBOL_MAP: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();

fn get_symbol_map() -> &'static HashMap<&'static str, &'static str> {
    SYMBOL_MAP.get_or_init(|| {
        let mut m = HashMap::new();
        // Core cleanups that are very unlikely to harm
        m.insert("“", "\"");
        m.insert("”", "\"");
        m.insert("‘", "'");
        m.insert("’", "'");
        m.insert("…", "...");
        // Remove zero-width spaces and other invisible formatters
        m.insert("\u{200B}", ""); // zero width space
        m.insert("\u{200C}", ""); // zero width non-joiner
        m.insert("\u{200D}", ""); // zero width joiner
        m.insert("\u{FEFF}", ""); // zero width no-break space
        m
    })
}

// Conservative chunk size to start - we can tune this
const MAX_CHUNK_BYTES: usize = 1000;

#[derive(Clone, Debug)]
pub struct TextChunk {
    pub text: String,
    pub should_pause_after: bool,
}

pub fn preprocess_text(input: &str) -> Vec<TextChunk> {
    // 1. Clean the text first
    let cleaned = clean_text(input);

    // 2. Split into sentences, keeping track of natural pause points
    split_into_chunks(&cleaned)
}

fn clean_text(text: &str) -> String {
    let mut result = text.trim().to_string();

    // Apply symbol mapping
    for (from, to) in get_symbol_map().iter() {
        result = result.replace(from, to);
    }

    // Collapse multiple spaces
    result = result.split_whitespace().collect::<Vec<_>>().join(" ");

    // Collapse multiple periods and multiple commas
    result = result.replace("....", "...");
    result = result.replace("...", ".");
    result = result.replace("..", ".");
    result = result.replace(",,", ",");

    result
}

fn split_into_chunks(text: &str) -> Vec<TextChunk> {
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut current_bytes = 0;

    // First split on sentence boundaries
    let sentences = text.split_inclusive(&['.', '!', '?'][..]);

    for sentence in sentences {
        let sentence_bytes = sentence.len();

        // If this single sentence is too big, we'll need to split it further
        if sentence_bytes > MAX_CHUNK_BYTES {
            // First flush any existing chunk
            if !current_chunk.is_empty() {
                chunks.push(TextChunk {
                    text: current_chunk,
                    should_pause_after: true,
                });
                current_chunk = String::new();
                current_bytes = 0;
            }

            // Split long sentence on commas
            let comma_parts: Vec<&str> = sentence.split(',').collect();
            for (i, part) in comma_parts.iter().enumerate() {
                let is_last = i == comma_parts.len() - 1;
                let mut part = part.to_string();
                if !is_last {
                    part.push(',');
                }

                // If even comma-split part is too big, split on spaces
                if part.len() > MAX_CHUNK_BYTES {
                    let words: Vec<&str> = part.split_whitespace().collect();
                    let mut sub_chunk = String::new();
                    let mut sub_bytes = 0;

                    for word in words {
                        if sub_bytes + word.len() + 1 > MAX_CHUNK_BYTES {
                            if !sub_chunk.is_empty() {
                                chunks.push(TextChunk {
                                    text: sub_chunk,
                                    should_pause_after: false,
                                });
                            }
                            sub_chunk = word.to_string();
                            sub_bytes = word.len();
                        } else {
                            if !sub_chunk.is_empty() {
                                sub_chunk.push(' ');
                                sub_bytes += 1;
                            }
                            sub_chunk.push_str(word);
                            sub_bytes += word.len();
                        }
                    }

                    if !sub_chunk.is_empty() {
                        chunks.push(TextChunk {
                            text: sub_chunk,
                            should_pause_after: is_last,
                        });
                    }
                } else {
                    chunks.push(TextChunk {
                        text: part,
                        should_pause_after: is_last,
                    });
                }
            }
        } else if current_bytes + sentence_bytes > MAX_CHUNK_BYTES {
            // Current chunk would be too big, flush it and start new one
            if !current_chunk.is_empty() {
                chunks.push(TextChunk {
                    text: current_chunk,
                    should_pause_after: true,
                });
            }
            current_chunk = sentence.to_string();
            current_bytes = sentence_bytes;
        } else {
            // Add to current chunk
            current_chunk.push_str(sentence);
            current_bytes += sentence_bytes;
        }
    }

    // Don't forget any remaining text
    if !current_chunk.is_empty() {
        chunks.push(TextChunk {
            text: current_chunk,
            should_pause_after: true,
        });
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_cleaning() {
        let input = "Hello... World";
        let cleaned = clean_text(input);
        assert_eq!(cleaned, "Hello. World");
    }

    #[test]
    fn test_smart_quotes() {
        let input = "“Hello” and ‘Hi’";
        let cleaned = clean_text(input);
        assert_eq!(cleaned, "\"Hello\" and 'Hi'");
    }

    #[test]
    fn test_chunk_splitting() {
        let input = "This is a short sentence. This is another one! What about this? Yes indeed.";
        let chunks = preprocess_text(input);
        assert_eq!(chunks.len(), 1); // Should fit in one chunk

        // Create a sentence that's definitely longer than MAX_CHUNK_BYTES
        let long_words =
            "supercalifragilisticexpialidociousasdfasdfasdfasdfasdfasdfasdfasdf ".repeat(20);
        let long_sentence = format!("{}.", long_words);
        assert!(
            long_sentence.len() > MAX_CHUNK_BYTES,
            "Test sentence length {} should exceed MAX_CHUNK_BYTES {}",
            long_sentence.len(),
            MAX_CHUNK_BYTES
        );

        let chunks = preprocess_text(&long_sentence);
        assert!(
            chunks.len() > 1,
            "Expected multiple chunks for text of length {}",
            long_sentence.len()
        );

        // Verify pause flags
        for (i, chunk) in chunks.iter().enumerate() {
            if i == chunks.len() - 1 {
                assert!(chunk.should_pause_after);
            }
        }
    }

    #[test]
    fn test_chunk_sizes() {
        // Verify no chunk exceeds MAX_CHUNK_BYTES
        let long_words = "supercalifragilisticexpialidocious ".repeat(20);
        let long_sentence = format!("{}.", long_words);

        let chunks = preprocess_text(&long_sentence);
        for chunk in &chunks {
            assert!(
                chunk.text.len() <= MAX_CHUNK_BYTES,
                "Chunk length {} exceeds MAX_CHUNK_BYTES {}",
                chunk.text.len(),
                MAX_CHUNK_BYTES
            );
        }
    }
}
