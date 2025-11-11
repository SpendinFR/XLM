use once_cell::sync::Lazy;
use regex::Regex;

pub fn slugify(input: &str) -> String {
    static RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[^a-z0-9]+").unwrap());
    let lowercase = input.to_lowercase();
    let trimmed = lowercase.trim();
    let replaced = RE.replace_all(trimmed, "-");
    let slug = replaced.trim_matches('-');
    if slug.is_empty() {
        "concept".to_string()
    } else {
        slug.to_string()
    }
}
