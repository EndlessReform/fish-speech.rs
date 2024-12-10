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
                                  // Japanese-specific punctuation normalization

        m.insert("。", "."); // Japanese period
        m.insert("、", ", "); // Comma - note the space after!
        m.insert("！", "!"); // Japanese exclamation
        m.insert("？", "?"); // Japanese question mark
        m.insert("「", "\""); // Japanese opening quote
        m.insert("」", "\""); // Japanese closing quote
        m.insert("『", "\""); // Japanese opening double quote
        m.insert("』", "\""); // Japanese closing double quote
        m.insert("・", ""); // Nakaguro (middle dot) - remove it like in original
        m.insert("：", ","); // Japanese colon to comma (following original)
        m.insert("；", ","); // Japanese semicolon to comma (following original)
        m.insert("（", ""); // Japanese opening parenthesis - remove
        m.insert("）", ""); // Japanese closing parenthesis - remove
        m.insert("【", ""); // Japanese opening bracket - remove
        m.insert("】", ""); // Japanese closing bracket - remove
        m
    })
}

// Constants at the top of the file
const MAX_CHUNK_BYTES: usize = 500;
const MIN_CHUNK_BYTES: usize = 100;
const PREFERRED_CHUNK_BYTES: usize = 250;

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

    // Format numbers over 1000 with commas (without adding spaces)
    let mut formatted = String::with_capacity(result.len());
    let mut num_buffer = String::new();

    for c in result.chars() {
        if c.is_numeric() {
            num_buffer.push(c);
        } else {
            if !num_buffer.is_empty() {
                // Format number if it's over 1000
                if num_buffer.len() > 3 {
                    let num: u64 = num_buffer.parse().unwrap_or(0);
                    let formatted_num = format!("{num:0}");
                    // Insert commas manually
                    let mut with_commas = String::new();
                    let num_str = formatted_num.chars().rev().collect::<Vec<_>>();
                    for (i, digit) in num_str.iter().enumerate() {
                        if i > 0 && i % 3 == 0 {
                            with_commas.push(',');
                        }
                        with_commas.push(*digit);
                    }
                    formatted.push_str(&with_commas.chars().rev().collect::<String>());
                } else {
                    formatted.push_str(&num_buffer);
                }
                num_buffer.clear();
            }
            formatted.push(c);
        }
    }

    // Handle any remaining number in buffer
    if !num_buffer.is_empty() {
        if num_buffer.len() > 3 {
            let num: u64 = num_buffer.parse().unwrap_or(0);
            let formatted_num = format!("{num:0}");
            let mut with_commas = String::new();
            let num_str = formatted_num.chars().rev().collect::<Vec<_>>();
            for (i, digit) in num_str.iter().enumerate() {
                if i > 0 && i % 3 == 0 {
                    with_commas.push(',');
                }
                with_commas.push(*digit);
            }
            formatted.push_str(&with_commas.chars().rev().collect::<String>());
        } else {
            formatted.push_str(&num_buffer);
        }
    }

    result = formatted;

    // Rest of the function remains the same...
    result = result.split_whitespace().collect::<Vec<_>>().join(" ");

    result = result.replace("....", "...");
    result = result.replace("...", ".");
    result = result.replace("..", ".");
    result = result.replace(",,", ",");

    result = result.replace(". ", ". ");
    result = result.replace("! ", "! ");
    result = result.replace("? ", "? ");

    let mut formatted = String::with_capacity(result.len());
    let chars: Vec<char> = result.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        formatted.push(chars[i]);
        if chars[i] == ',' && i + 1 < chars.len() {
            if !chars[i - 1].is_numeric() || !chars[i + 1].is_numeric() {
                formatted.push(' ');
            }
        }
        i += 1;
    }
    result = formatted;

    result = result.split_whitespace().collect::<Vec<_>>().join(" ");

    result.trim().to_string() + " "
}

fn split_into_chunks(text: &str) -> Vec<TextChunk> {
    let mut chunks = Vec::new();

    // First pass: collect sentences and their sizes
    let sentences: Vec<&str> = text.split_inclusive(&['.', '!', '?'][..]).collect();
    if sentences.is_empty() {
        return chunks;
    }

    // let flush_chunk = |chunk: String, force_pause: bool| -> Option<String> {
    //     if chunk.is_empty() {
    //         return None;
    //     }

    //     // Only flush if we're over MIN_CHUNK_BYTES or it's the very last bit
    //     if chunk.len() >= MIN_CHUNK_BYTES || force_pause {
    //         chunks.push(TextChunk {
    //             text: chunk,
    //             should_pause_after: force_pause,
    //         });
    //         None
    //     } else {
    //         // Return the chunk to be combined with the next one
    //         Some(chunk)
    //     }
    // };

    let mut pending_small_chunk: Option<String> = None;

    // Process sentences with lookahead
    let mut i = 0;
    while i < sentences.len() {
        let mut combined = pending_small_chunk.unwrap_or_default();
        let mut combined_bytes = combined.len();
        let mut sentences_in_chunk = 0;

        // Look ahead and combine sentences until we hit our targets
        while i + sentences_in_chunk < sentences.len() {
            let next_sentence = sentences[i + sentences_in_chunk];
            let next_bytes = next_sentence.len();

            // If adding this sentence would exceed MAX_CHUNK_BYTES
            if combined_bytes + next_bytes > MAX_CHUNK_BYTES {
                if combined_bytes >= MIN_CHUNK_BYTES || i + sentences_in_chunk == sentences.len() {
                    break;
                }
                // If we're under MIN_CHUNK_BYTES, try to add it anyway
            }

            combined.push_str(next_sentence);
            combined_bytes += next_bytes;
            sentences_in_chunk += 1;

            // Only break if we're over MIN_CHUNK_BYTES and near PREFERRED_CHUNK_BYTES
            if combined_bytes >= MIN_CHUNK_BYTES && combined_bytes >= PREFERRED_CHUNK_BYTES {
                break;
            }
        }

        // Decide whether to flush or keep pending
        if combined_bytes < MIN_CHUNK_BYTES && i + sentences_in_chunk < sentences.len() {
            pending_small_chunk = Some(combined);
        } else {
            pending_small_chunk = None;
            chunks.push(TextChunk {
                text: combined,
                should_pause_after: i + sentences_in_chunk >= sentences.len(),
            });
        }

        i += sentences_in_chunk;
    }

    // Handle any remaining pending chunk
    if let Some(last_chunk) = pending_small_chunk {
        chunks.push(TextChunk {
            text: last_chunk,
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

    #[test]
    fn test_japanese_punctuation() {
        let input = "これは「テスト」です。かっこ（テスト）は？はい！";
        let cleaned = clean_text(input);
        assert_eq!(cleaned, "これは\"テスト\"です. かっこテストは? はい!");
    }

    #[test]
    fn test_japanese_torture_cases() {
        let tests = vec![
            // Case 1: Mixed tiny and huge sentences
            ("こんにちは。とてもとても短い。超短。はい。この文は特別長くて、色々な読点を含んでおり、実際の運用でよく見られるような、アナウンサーが早口で読み上げるような、そういった感じの文章なのです。短。はい。", "Mixed tiny/huge"),

            // Case 2: Pathologically long station announcement
            ("まもなく電車が参ります。危険ですので黄色い線の後ろまでお下がりください。この電車は、JR埼京線、新宿・渋谷方面行きです。停車駅は、池袋、新宿、渋谷、恵比寿、大崎、大井町、天王洲アイル、品川シーサイド、東京テレポート、台場、青海、国際展示場、東雲、新木場、まもなく到着いたします。この電車は、10両編成です。この電車の次は、各駅停車、東京・上野方面行きが参ります。車内が大変混雑しておりますので、押し合わないようご注意ください。", "Long station list"),

            // Case 3: Lots of tiny fragments with quotes and parentheses
            ("「はい」。『はい』。（はい）。【はい】。「こんにちは」。（お待たせしました）。『ありがとうございます』。短。短。短。", "Many fragments"),

            // Case 4: Numbers and times mixed with text
            ("まもなく13時35分発の電車が、2番線に参ります。この電車は、10両編成の通勤特急です。1号車から3号車までは、指定席です。4号車と5号車は、女性専用車両です。6号車から10号車までは、自由席です。", "Numbers and times"),

            // Case 5: Mix of super-long and normal sentences with nested punctuation
            ("この電車は、各駅停車（東京・上野方面行き【所要時間：約45分】）です。普通。はい。センテンス・アンド・センシビリティー（「知性と感性」）は、ジェーン・オースティンの小説です。この電車は、（途中駅、赤羽、川口、浦和、さいたま新都心、与野、大宮、を経由いたしまして）、最終的には、大変申し訳ございませんが、運転時刻の関係で、大幅に遅れる可能性がございます。", "Mixed with nesting"),

            // Case 6: Repetitive announcements with slight variations
            ("お待たせいたしました。お待たせいたしました。お待たせいたしました。この電車は、まもなく発車いたします。この電車は、まもなく発車いたします。ドアが閉まります。ご注意ください。ドアが閉まります。ご注意ください。", "Repetitive"),

            // Case 7: Edge case with no proper sentence endings
            ("あのー、えーと、そうですね、はい、あのー、えーと、そうですね、はい、あのー、えーと、そうですね、はい", "No endings"),
        ];

        for (input, case_name) in tests {
            println!("\nTesting case: {}", case_name);

            let chunks = preprocess_text(input);

            println!("Input length: {} bytes", input.len());
            println!("Number of chunks: {}", chunks.len());

            for (i, chunk) in chunks.iter().enumerate() {
                println!("\nChunk {} ({}bytes):{}", i, chunk.text.len(), chunk.text);

                // Size assertions
                assert!(
                    chunk.text.len() <= MAX_CHUNK_BYTES,
                    "Chunk {} in case '{}' exceeds MAX_CHUNK_BYTES",
                    i,
                    case_name
                );

                // Unless it's the last tiny bit of a no-endings case
                if !(i == chunks.len() - 1 && case_name == "No endings") {
                    assert!(
                        chunk.text.len() >= MIN_CHUNK_BYTES,
                        "Chunk {} in case '{}' is smaller than MIN_CHUNK_BYTES",
                        i,
                        case_name
                    );
                }

                // Spacing consistency
                assert!(
                    !chunk.text.contains("  "),
                    "Chunk {} in case '{}' has double spaces",
                    i,
                    case_name
                );

                // Japanese punctuation conversion
                assert!(
                    !chunk.text.contains('。'),
                    "Chunk {} in case '{}' contains unconverted periods",
                    i,
                    case_name
                );
                assert!(
                    !chunk.text.contains('、'),
                    "Chunk {} in case '{}' contains unconverted commas",
                    i,
                    case_name
                );

                // Proper pause flags
                if i == chunks.len() - 1 {
                    assert!(
                        chunk.should_pause_after,
                        "Last chunk in case '{}' should have pause",
                        case_name
                    );
                }
            }
        }
    }

    #[test]
    fn test_content_preservation() {
        let test_cases = vec![
            // Classic election van speech with repetitions
            ("おはようございます! おはようございます! 自民党公認! 比例代表! クロードでございます! もう一度申し上げます! クロードでございます!",
             vec!["おはようございます", "自民党公認", "比例代表", "クロード", "もう一度申し上げます"]),

            // Numbers and formatting
            ("1番! 景気対策! 2番! 子育て支援! 3番! 年金改革! 4番! 環境政策! 5番! 地域活性化!",
             vec!["1番", "2番", "3番", "4番", "5番", "景気対策", "子育て支援", "年金改革", "環境政策", "地域活性化"]),

            // Long numbers with commas
            ("1,000円が2,000円に! 5,000円が10,000円に!",
             vec!["1,000", "2,000", "5,000", "10,000"]),
        ];

        for (input, key_phrases) in test_cases {
            let chunks = preprocess_text(input);

            // Combine all chunks into one string for easy searching
            let processed = chunks
                .iter()
                .map(|chunk| chunk.text.as_str())
                .collect::<String>();

            // Check each key phrase is present
            for phrase in key_phrases {
                assert!(
                    processed.contains(phrase),
                    "Missing phrase '{}' in processed text.\nInput: {}\nProcessed: {}",
                    phrase,
                    input,
                    processed
                );
            }

            // Count exclamation marks - should be the same before and after
            let input_exclam_count = input.matches('!').count();
            let processed_exclam_count = processed.matches('!').count();
            assert_eq!(input_exclam_count, processed_exclam_count,
                "Exclamation mark count mismatch.\nInput had {}, processed had {}\nInput: {}\nProcessed: {}",
                input_exclam_count, processed_exclam_count, input, processed);

            // Verify all numbers are preserved
            let input_numbers: Vec<&str> = input
                .split(|c: char| !c.is_numeric() && c != ',')
                .filter(|s| !s.is_empty() && s.chars().any(|c| c.is_numeric()))
                .collect();
            for num in input_numbers {
                assert!(
                    processed.contains(num),
                    "Missing number '{}' in processed text.\nInput: {}\nProcessed: {}",
                    num,
                    input,
                    processed
                );
            }

            // Check total character count (excluding spaces and normalized punctuation)
            let input_chars = input
                .chars()
                .filter(|c| !c.is_whitespace() && *c != '!' && *c != '.' && *c != ',')
                .collect::<String>();
            let processed_chars = processed
                .chars()
                .filter(|c| !c.is_whitespace() && *c != '!' && *c != '.' && *c != ',')
                .collect::<String>();
            assert_eq!(input_chars.len(), processed_chars.len(),
                "Character count mismatch (excluding punctuation and spaces).\nInput: {}\nProcessed: {}",
                input_chars, processed_chars);
        }
    }
}
