use super::clean::clean_text;
use regex::Regex;
use std::collections::HashSet;
use std::sync::OnceLock;

static FLOAT_PROTECT_RE: OnceLock<Regex> = OnceLock::new();
static FLOAT_UNPROTECT_RE: OnceLock<Regex> = OnceLock::new();

fn utf_8_len(text: &str) -> usize {
    text.len()
}

/// Splits texts based on specified split characters.
/// If a text's byte length is less than or equal to `length`, it is yielded as-is.
/// Otherwise, it is split whenever a character in `splits` is encountered.
fn break_text<I>(texts: I, splits: &HashSet<char>) -> Vec<String>
where
    I: Iterator<Item = String>,
{
    let mut result = Vec::new();

    for text in texts {
        if utf_8_len(&text) == 0 {
            continue;
        }

        if utf_8_len(&text) <= usize::MAX {
            // Since we don't have a length parameter here, we proceed to split based on splits.
            let mut current = String::new();
            for c in text.chars() {
                current.push(c);
                if splits.contains(&c) {
                    if !current.is_empty() {
                        result.push(current.clone());
                        current.clear();
                    }
                }
            }
            if !current.is_empty() {
                result.push(current);
            }
        } else {
            // This branch is unlikely since we're using usize::MAX, but included for completeness.
            result.push(text);
        }
    }

    result
}

/// Splits texts based on byte length.
/// If a text's byte length is less than or equal to `length`, it is yielded as-is.
/// Otherwise, it is split to ensure each segment's byte length does not exceed `length`.
fn break_text_by_length<I>(texts: I, length: usize) -> Vec<String>
where
    I: Iterator<Item = String>,
{
    let mut result = Vec::new();

    for text in texts {
        if utf_8_len(&text) <= length {
            result.push(text);
            continue;
        }

        let mut current = String::new();
        for c in text.chars() {
            let char_len = c.len_utf8();
            if utf_8_len(&current) + char_len > length {
                if !current.is_empty() {
                    result.push(current.clone());
                    current.clear();
                }
            }
            current.push(c);
        }
        if !current.is_empty() {
            result.push(current);
        }
    }

    result
}

/// Adds a cleaned segment to the segments vector if it meets the criteria.
fn add_cleaned(curr: &str, segments: &mut Vec<String>) {
    let trimmed = curr.trim();
    if !trimmed.is_empty()
        && !trimmed
            .chars()
            .all(|c| c.is_whitespace() || c.is_ascii_punctuation())
    {
        segments.push(trimmed.to_string());
    }
}

/// Protects floating-point numbers in the text to prevent splitting.
/// Converts patterns like "3.14" to "<3_f_14>".
fn protect_float(text: &str, float_re: &Regex) -> String {
    float_re.replace_all(text, "<$1_f_$2>").to_string()
}

/// Unprotects floating-point numbers in the text.
/// Converts patterns like "<3_f_14>" back to "3.14".
fn unprotect_float(text: &str, float_re: &Regex) -> String {
    float_re.replace_all(text, "$1.$2").to_string()
}

/// Splits the text based on the specified rules.
pub fn split_text(text: &str, length: usize) -> Vec<String> {
    // Initialize regular expressions using lazy_static for efficiency.
    let float_protect_re = FLOAT_PROTECT_RE
        .get_or_init(|| Regex::new(r"(\d+)\.(\d+)").expect("Failed to compile FLOAT_PROTECT_RE"));

    let float_unprotect_re = FLOAT_UNPROTECT_RE.get_or_init(|| {
        Regex::new(r"<(\d+)_f_(\d+)>").expect("Failed to compile FLOAT_UNPROTECT_RE")
    });

    let text = clean_text(text);
    let texts = vec![protect_float(&text, &float_protect_re)];

    // Step 3: Break text at ".", "!", "?".
    let splits_sentence: HashSet<char> = ['.', '!', '?'].iter().cloned().collect();
    let texts = break_text(texts.into_iter(), &splits_sentence);

    // Step 4: Unprotect floats.
    let texts = texts
        .iter()
        .map(|s| unprotect_float(s, &float_unprotect_re))
        .collect::<Vec<String>>();

    // Step 5: Break text at ",".
    let splits_comma: HashSet<char> = [','].iter().cloned().collect();
    let texts = break_text(texts.into_iter(), &splits_comma);

    // Step 6: Break text at " ".
    let splits_space: HashSet<char> = [' '].iter().cloned().collect();
    let texts = break_text(texts.into_iter(), &splits_space);

    // Step 7: Break text by length.
    let texts = break_text_by_length(texts.into_iter(), length);

    // Step 8: Merge texts into segments with length <= `length`.
    let mut segments = Vec::new();
    let mut current = String::new();

    for text in texts {
        if utf_8_len(&current) + utf_8_len(&text) <= length {
            current.push_str(&text);
        } else {
            add_cleaned(&current, &mut segments);
            current = text;
        }
    }

    if !current.is_empty() {
        add_cleaned(&current, &mut segments);
    }

    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_utf_8_len() {
        assert_eq!(utf_8_len("a"), 1);
        assert_eq!(utf_8_len("你好"), 6);
    }

    #[test]
    fn test_split_text_basic() {
        let text = "This is a test sentence. This is another test sentence. And a third one.";
        let expected = vec![
            "This is a test sentence.",
            "This is another test sentence. And a third one.",
        ];
        assert_eq!(split_text(text, 50), expected);
    }

    #[test]
    fn test_split_text_with_float() {
        let text = "a,aaaaaa3.14";
        let expected = vec!["a,", "aaaaaa3.14"];
        assert_eq!(split_text(text, 10), expected);
    }

    #[test]
    fn test_split_text_empty() {
        let text = "   ";
        let expected: Vec<String> = vec![];
        assert_eq!(split_text(text, 10), expected);
    }

    #[test]
    fn test_split_text_single_char() {
        let text = "a";
        let expected = vec!["a"];
        assert_eq!(split_text(text, 10), expected);
    }

    #[test]
    fn test_split_text_commas() {
        let text = "This is a test sentence with only commas, and no dots, and no exclamation marks, and no question marks, and no newlines.";
        let expected = vec![
            "This is a test sentence with only commas,",
            "and no dots, and no exclamation marks,",
            "and no question marks, and no newlines.",
        ];
        assert_eq!(split_text(text, 50), expected);
    }

    #[test]
    fn test_split_text_mixed() {
        let text = "This is a test sentence This is a test sentence This is a test sentence. This is a test sentence, This is a test sentence, This is a test sentence.";
        let expected = vec![
            "This is a test sentence This is a test sentence",
            "This is a test sentence. This is a test sentence,",
            "This is a test sentence, This is a test sentence.",
        ];
        assert_eq!(split_text(text, 50), expected);
    }

    #[test]
    fn test_split_text_chinese() {
        let text = "这是一段很长的中文文本,而且没有句号,也没有感叹号,也没有问号,也没有换行符。";
        let expected = vec![
            "这是一段很长的中文文本,",
            "而且没有句号,也没有感叹号,",
            "也没有问号,也没有换行符.",
        ];
        assert_eq!(split_text(text, 50), expected);
    }

    #[test]
    fn test_protect_unprotect_float() {
        let original = "The value is 3.14159.";
        let protected = protect_float(original, &Regex::new(r"(\d+)\.(\d+)").unwrap());
        let expected_protected = "The value is <3_f_14159>.";
        assert_eq!(protected, expected_protected);

        let unprotected = unprotect_float(&protected, &Regex::new(r"<(\d+)_f_(\d+)>").unwrap());
        assert_eq!(unprotected, original);
    }

    #[test]
    fn test_add_cleaned() {
        let mut segments = Vec::new();
        add_cleaned("  Hello, World!  ", &mut segments);
        add_cleaned("   ", &mut segments);
        add_cleaned("!!!", &mut segments);
        add_cleaned("Rust is awesome.", &mut segments);
        assert_eq!(
            segments,
            vec!["Hello, World!".to_string(), "Rust is awesome.".to_string()]
        );
    }
}
