//! Data bindings: the named values a program reads, from Rust or from JSON.

use std::collections::BTreeMap;

use super::{ProgramError, Value};

/// Named values bound as variables when a program is
/// [compiled](super::Program::compile): the observed data, and any scalar
/// settings a host wants to pass in.
///
/// ```rust
/// use fugue::program::{Data, Value};
///
/// let from_rust = Data::new()
///     .with("y", vec![1.3, 0.7, 2.1])
///     .with("n", 3)
///     .with("strict", true);
/// let from_json = Data::from_json(r#"{"y": [1.3, 0.7, 2.1], "n": 3, "strict": true}"#).unwrap();
/// assert_eq!(from_rust, from_json);
/// assert_eq!(from_json.get("n"), Some(&Value::Int(3)));
///
/// // A bare array binds `data`, as the playground has always read it.
/// let bare = Data::from_json("[1, 0, 1]").unwrap();
/// assert_eq!(bare.get("data").and_then(Value::as_array), Some(&[1.0, 0.0, 1.0][..]));
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Data {
    vars: BTreeMap<String, Value>,
}

impl Data {
    /// No bindings.
    pub fn new() -> Data {
        Data::default()
    }

    /// Read bindings from JSON: an object whose values are numbers, booleans,
    /// or arrays of numbers/booleans; a bare array (bound to `data`); or an
    /// empty string / `null` for none.
    ///
    /// Arrays become [`Value::Arr`] (booleans read as `1.0`/`0.0`). A scalar
    /// integer becomes [`Value::Int`] (or [`Value::U64`] above `i64::MAX`),
    /// any other number [`Value::F64`], a boolean [`Value::Bool`].
    pub fn from_json(json: &str) -> Result<Data, ProgramError> {
        let trimmed = json.trim();
        let mut data = Data::new();
        if trimmed.is_empty() || trimmed == "null" {
            return Ok(data);
        }
        let parsed: serde_json::Value = serde_json::from_str(trimmed)
            .map_err(|e| ProgramError::Data(format!("data is not valid JSON: {e}")))?;
        match parsed {
            serde_json::Value::Array(items) => {
                data.vars.insert("data".to_string(), array("data", &items)?);
            }
            serde_json::Value::Object(map) => {
                for (name, v) in map {
                    let value = match &v {
                        serde_json::Value::Array(items) => array(&name, items)?,
                        serde_json::Value::Bool(b) => Value::Bool(*b),
                        serde_json::Value::Number(n) => {
                            if let Some(i) = n.as_i64() {
                                Value::Int(i)
                            } else if let Some(u) = n.as_u64() {
                                Value::U64(u)
                            } else {
                                Value::F64(n.as_f64().unwrap_or(f64::NAN))
                            }
                        }
                        _ => {
                            return Err(ProgramError::Data(format!(
                                "data `{name}` must be a number, a boolean, or an array of them"
                            )))
                        }
                    };
                    data.vars.insert(name, value);
                }
            }
            _ => {
                return Err(ProgramError::Data(
                    "data must be a JSON object (of arrays or scalars) or a bare array".to_string(),
                ))
            }
        }
        Ok(data)
    }

    /// These bindings plus `name` bound to `value` (builder style).
    pub fn with(mut self, name: impl Into<String>, value: impl Into<Value>) -> Data {
        self.insert(name, value);
        self
    }

    /// Bind `name` to `value`, returning the value it replaces.
    pub fn insert(&mut self, name: impl Into<String>, value: impl Into<Value>) -> Option<Value> {
        self.vars.insert(name.into(), value.into())
    }

    /// The value bound to `name`.
    pub fn get(&self, name: &str) -> Option<&Value> {
        self.vars.get(name)
    }

    /// The bindings, in name order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.vars.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// The number of bindings.
    pub fn len(&self) -> usize {
        self.vars.len()
    }

    /// Whether there are no bindings.
    pub fn is_empty(&self) -> bool {
        self.vars.is_empty()
    }
}

fn array(name: &str, items: &[serde_json::Value]) -> Result<Value, ProgramError> {
    items
        .iter()
        .map(|x| {
            x.as_f64()
                .or(x.as_bool().map(|b| if b { 1.0 } else { 0.0 }))
                .ok_or_else(|| {
                    ProgramError::Data(format!("data array `{name}` must hold numbers or booleans"))
                })
        })
        .collect::<Result<Vec<f64>, _>>()
        .map(Value::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_shapes() {
        assert!(Data::from_json("").unwrap().is_empty());
        assert!(Data::from_json("  null ").unwrap().is_empty());
        let d = Data::from_json(r#"{"x": [1, true, 2.5], "k": 18446744073709551615, "f": 0.5}"#)
            .unwrap();
        assert_eq!(d.get("x"), Some(&Value::from(vec![1.0, 1.0, 2.5])));
        assert_eq!(d.get("k"), Some(&Value::U64(u64::MAX)));
        assert_eq!(d.get("f"), Some(&Value::F64(0.5)));
        assert_eq!(d.len(), 3);
        assert_eq!(
            d.iter().map(|(k, _)| k).collect::<Vec<_>>(),
            ["f", "k", "x"]
        );
        for bad in [
            "{",
            "3",
            r#"{"x": "s"}"#,
            r#"{"x": [null]}"#,
            r#"{"x": [[1]]}"#,
        ] {
            assert!(
                matches!(Data::from_json(bad), Err(ProgramError::Data(_))),
                "{bad}"
            );
        }
    }
}
