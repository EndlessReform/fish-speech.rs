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
        m.insert("«", "\""); // French opening quote
        m.insert("»", "\""); // French closing quote
        m.insert(" « ", "\""); // French quotes with spaces
        m.insert(" » ", "\"");

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
        m.insert("・", ""); // Nakaguro (middle dot)
        m.insert("：", ","); // Japanese colon to comma
        m.insert("；", ","); // Japanese semicolon to comma
        m.insert("（", ""); // Japanese opening parenthesis
        m.insert("）", ""); // Japanese closing parenthesis
        m.insert("【", ""); // Japanese opening bracket
        m.insert("】", ""); // Japanese closing bracket
        m
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Script {
    Chinese,  // Primarily hanzi
    Japanese, // Mix of kanji and kana
    Korean,   // Primarily hangul
    Latin,    // Everything else
}

fn get_thresholds(script: &Script) -> (usize, usize) {
    match script {
        // (combine_threshold, split_threshold)
        Script::Chinese => (30, 100),  // Chinese is dense
        Script::Japanese => (45, 150), // Japanese has particles
        Script::Korean => (40, 120),   // Similar to Japanese
        Script::Latin => (150, 400),   // Much longer before we panic
    }
}

fn clean_text(text: &str) -> String {
    let mut result = text.trim().to_string();

    // Apply symbol mapping
    for (from, to) in get_symbol_map().iter() {
        result = result.replace(from, to);
    }

    // Strip emoji
    result = result
        .chars()
        .filter(|&c| !('\u{1F300}'..='\u{1F9FF}').contains(&c))
        .collect();

    // Normalize dash-like things to em-dash
    result = result
        .replace(" - ", "—")
        .replace("--", "—")
        .replace(" – ", "—");

    // Normalize multiple punctuation
    result = result
        .replace("....", ".")
        .replace("...", ".")
        .replace("..", ".")
        .replace(",,", ",");

    // Ensure single spaces between words and after punctuation
    result.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn is_hanzi(c: char) -> bool {
    ('\u{4E00}'..='\u{9FFF}').contains(&c) // Basic CJK unified
}

fn is_kana(c: char) -> bool {
    ('\u{3040}'..='\u{309F}').contains(&c) ||  // Hiragana
    ('\u{30A0}'..='\u{30FF}').contains(&c) // Katakana
}

fn is_hangul(c: char) -> bool {
    ('\u{AC00}'..='\u{D7AF}').contains(&c) // Hangul syllables
}

fn detect_script(text: &str) -> Script {
    let chars: Vec<_> = text.chars().collect();
    if chars.is_empty() {
        return Script::Latin;
    }

    let total = chars.len() as f32;
    let hanzi = chars.iter().filter(|c| is_hanzi(**c)).count() as f32;
    let kana = chars.iter().filter(|c| is_kana(**c)).count() as f32;
    let hangul = chars.iter().filter(|c| is_hangul(**c)).count() as f32;

    if hanzi / total > 0.5 && kana / total < 0.1 {
        Script::Chinese
    } else if kana / total > 0.2 || (hanzi / total > 0.2 && kana / total > 0.1) {
        Script::Japanese
    } else if hangul / total > 0.3 {
        Script::Korean
    } else {
        Script::Latin
    }
}

pub fn preprocess_text(text: &str) -> Vec<String> {
    let text = clean_text(text);
    let script = detect_script(&text);

    // Split on major sentence boundaries first
    let sentences: Vec<&str> = text
        .split_inclusive(&['.', '!', '?'][..])
        .filter(|s| !s.trim().is_empty())
        .collect();

    if sentences.is_empty() {
        return vec![];
    }

    let mut chunks = Vec::new();

    // First chunk gets base thresholds
    let (combine_threshold, split_threshold) = get_thresholds(&script);
    println!(
        "Processing text with script {:?}, initial thresholds: combine={}, split={}",
        script, combine_threshold, split_threshold
    );

    // Always output first sentence ASAP for TTFT
    let first = sentences[0];
    if first.chars().count() <= split_threshold {
        chunks.push(first.to_string());
    } else {
        // If first sentence is huge, reluctantly split on commas
        for piece in first.split_inclusive(&[',', '，', '、'][..]) {
            if !piece.trim().is_empty() {
                chunks.push(piece.trim().to_string());
            }
        }
    }

    // Process remaining sentences with progressive thresholds
    let mut current = String::new();
    let mut chunk_index = chunks.len(); // Start counting from where we are

    for sentence in &sentences[1..] {
        // Get progressive thresholds based on how many chunks we've output
        let multiplier = (1.0 + (chunk_index as f32 * 0.2)).min(2.0);
        let (combine_threshold, split_threshold) = (
            (combine_threshold as f32 * multiplier) as usize,
            (split_threshold as f32 * multiplier) as usize,
        );

        let sentence_chars = sentence.trim().chars().count();

        // If this single sentence exceeds current split threshold,
        // reluctantly split on commas
        if sentence_chars > split_threshold {
            // First flush any pending content
            if !current.is_empty() {
                chunks.push(current.trim().to_string());
                current.clear();
                chunk_index += 1;
            }

            // Split the long sentence
            for piece in sentence.split_inclusive(&[',', '，', '、'][..]) {
                if !piece.trim().is_empty() {
                    chunks.push(piece.trim().to_string());
                    chunk_index += 1;
                }
            }
            continue;
        }

        // Try to combine short sentences up to current combine_threshold
        if !current.is_empty() && (current.chars().count() + sentence_chars > combine_threshold) {
            chunks.push(current.trim().to_string());
            chunk_index += 1;
            current.clear();
        }

        if current.is_empty() {
            current = sentence.trim().to_string();
        } else {
            current.push(' ');
            current.push_str(sentence.trim());
        }
    }

    // Don't forget last chunk
    if !current.is_empty() {
        chunks.push(current.trim().to_string());
    }

    println!("Split into {} chunks with progressive sizing", chunks.len());
    println!("Chunks:\n{:?}", chunks);
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_script_detection() {
        assert_eq!(detect_script("Hello world"), Script::Latin);
        assert_eq!(detect_script("私は日本語を話します"), Script::Japanese);
        assert_eq!(detect_script("我爱北京天安门"), Script::Chinese);
        assert_eq!(detect_script("안녕하세요"), Script::Korean);
        assert_eq!(detect_script("漢字とひらがな"), Script::Japanese);
    }

    #[test]
    fn test_text_cleaning() {
        let text = "Hello 👋 World! Testing—some « quotes » and。。。ellipses...";
        let cleaned = clean_text(text);
        assert!(!cleaned.contains('👋'));
        assert!(cleaned.contains('—'));
        assert!(cleaned.contains('"'));
        assert!(!cleaned.contains('«'));
        assert!(!cleaned.contains('»'));
        assert!(!cleaned.contains("..."));
    }

    #[test]
    fn test_mixed_scripts() {
        let text = "This is English. 这是中文。これは日本語です。";
        let chunks = preprocess_text(text);
        assert!(chunks.len() >= 3);
    }
}
