//! Tests for the enhanced error handling system.
//!
//! This test module demonstrates and validates the new error handling features:
//! - Error codes for programmatic handling  
//! - Rich error context with key-value pairs
//! - Error categorization and classification
//! - Helper macros for convenient error creation
//! - From trait implementations for common conversions

use fugue::*;

#[test]
fn test_error_codes_and_categories() {
    // Test parameter validation error codes
    let normal_err = Normal::new(f64::NAN, 1.0).unwrap_err();
    assert_eq!(normal_err.code(), ErrorCode::InvalidMean);
    assert_eq!(normal_err.category(), ErrorCategory::DistributionValidation);
    assert!(normal_err.is_validation_error());
    assert!(!normal_err.is_numerical_error());

    let uniform_err = Uniform::new(5.0, 1.0).unwrap_err();
    assert_eq!(uniform_err.code(), ErrorCode::InvalidRange);
    assert_eq!(
        uniform_err.category(),
        ErrorCategory::DistributionValidation
    );

    // Test that error messages include error codes
    let error_msg = format!("{}", normal_err);
    assert!(error_msg.contains("[100]")); // InvalidMean = 100
}

#[test]
fn test_error_context_information() {
    // Test that errors include contextual parameter information
    let gamma_err = Gamma::new(-1.0, 2.0).unwrap_err();

    let error_msg = format!("{}", gamma_err);
    assert!(error_msg.contains("shape=-1"));
    assert!(error_msg.contains("expected=> 0.0 and finite"));

    // Test categorical distribution with detailed context
    let cat_err = Categorical::new(vec![0.3, 0.8]).unwrap_err();
    let error_msg = format!("{}", cat_err);
    assert!(error_msg.contains("sum=1.100000"));
    assert!(error_msg.contains("expected=1.0"));
    assert!(error_msg.contains("tolerance=1e-6"));
}

#[test]
fn test_trace_error_handling() {
    let trace = Trace::default();

    // Test TraceAddressNotFound error
    let result = trace.get_f64_result(&addr!("missing"));
    assert!(result.is_err());

    let err = result.unwrap_err();
    assert_eq!(err.code(), ErrorCode::TraceAddressNotFound);
    assert_eq!(err.category(), ErrorCategory::TraceManipulation);

    let error_msg = format!("{}", err);
    assert!(error_msg.contains("[500]")); // TraceAddressNotFound = 500
    assert!(error_msg.contains("get_f64"));
    assert!(error_msg.contains("missing"));
}

#[test]
fn test_type_mismatch_errors() {
    let mut trace = Trace::default();
    trace.insert_choice(addr!("x"), ChoiceValue::Bool(true), 0.0);

    // Try to get boolean value as f64 - should fail with TypeMismatch
    let result = trace.get_f64_result(&addr!("x"));
    assert!(result.is_err());

    let err = result.unwrap_err();
    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.category(), ErrorCategory::TypeSystem);

    let error_msg = format!("{}", err);
    assert!(error_msg.contains("[600]")); // TypeMismatch = 600
    assert!(error_msg.contains("expected f64"));
    assert!(error_msg.contains("found bool"));
}

#[test]
fn test_error_helper_macros() {
    // Test invalid_params! macro
    let err = invalid_params!("TestDist", "test reason", InvalidMean);
    assert_eq!(err.code(), ErrorCode::InvalidMean);

    let error_msg = format!("{}", err);
    assert!(error_msg.contains("[100]"));
    assert!(error_msg.contains("TestDist"));
    assert!(error_msg.contains("test reason"));

    // Test invalid_params! macro with context
    let err_with_ctx = invalid_params!("TestDist", "test reason", InvalidMean,
        "param" => "value", "expected" => "different");

    let error_msg = format!("{}", err_with_ctx);
    assert!(error_msg.contains("param=value"));
    assert!(error_msg.contains("expected=different"));
}

#[test]
fn test_from_trait_conversions() {
    // Test conversion from string slice
    let err: FugueError = "Something went wrong".into();
    assert_eq!(err.code(), ErrorCode::ModelExecutionFailed);
    assert_eq!(err.category(), ErrorCategory::ModelExecution);

    // Test conversion from String
    let err: FugueError = String::from("Another error").into();
    assert_eq!(err.code(), ErrorCode::ModelExecutionFailed);
}

#[test]
fn test_error_classification() {
    let validation_err = Normal::new(0.0, -1.0).unwrap_err();
    assert!(validation_err.is_validation_error());
    assert!(!validation_err.is_numerical_error());
    assert!(!validation_err.is_recoverable());

    // Test a numerical error (through ParseFloatError conversion)
    let parse_result: Result<f64, _> = "not_a_number".parse();
    let numerical_err: FugueError = parse_result.unwrap_err().into();
    assert!(!numerical_err.is_validation_error());
    assert!(numerical_err.is_numerical_error());
}

#[test]
fn test_error_context_builder_methods() {
    let mut err =
        FugueError::invalid_parameters("TestDist", "invalid parameter", ErrorCode::InvalidMean);

    // Add context using builder methods
    err = err
        .with_context("param", "value")
        .with_context("suggestion", "use positive values")
        .with_source_location("test_file.rs", 42);

    let error_msg = format!("{}", err);
    assert!(error_msg.contains("param=value"));
    assert!(error_msg.contains("suggestion=use positive values"));
    assert!(error_msg.contains("(at test_file.rs:42)"));
}

#[test]
fn test_comprehensive_distribution_validation() {
    // Test all distributions return appropriate error codes

    // Normal distribution
    assert_eq!(
        Normal::new(f64::NAN, 1.0).unwrap_err().code(),
        ErrorCode::InvalidMean
    );
    assert_eq!(
        Normal::new(0.0, -1.0).unwrap_err().code(),
        ErrorCode::InvalidVariance
    );

    // Bernoulli distribution
    assert_eq!(
        Bernoulli::new(-0.5).unwrap_err().code(),
        ErrorCode::InvalidProbability
    );
    assert_eq!(
        Bernoulli::new(1.5).unwrap_err().code(),
        ErrorCode::InvalidProbability
    );

    // Beta distribution
    assert_eq!(
        Beta::new(-1.0, 2.0).unwrap_err().code(),
        ErrorCode::InvalidShape
    );
    assert_eq!(
        Beta::new(1.0, -2.0).unwrap_err().code(),
        ErrorCode::InvalidShape
    );

    // Gamma distribution
    assert_eq!(
        Gamma::new(-1.0, 2.0).unwrap_err().code(),
        ErrorCode::InvalidShape
    );
    assert_eq!(
        Gamma::new(1.0, -2.0).unwrap_err().code(),
        ErrorCode::InvalidRate
    );

    // Exponential distribution
    assert_eq!(
        Exponential::new(-1.0).unwrap_err().code(),
        ErrorCode::InvalidRate
    );

    // Uniform distribution
    assert_eq!(
        Uniform::new(5.0, 1.0).unwrap_err().code(),
        ErrorCode::InvalidRange
    );
    assert_eq!(
        Uniform::new(f64::NAN, 1.0).unwrap_err().code(),
        ErrorCode::InvalidRange
    );

    // Poisson distribution
    assert_eq!(
        Poisson::new(-1.0).unwrap_err().code(),
        ErrorCode::InvalidRate
    );

    // Categorical distribution
    assert_eq!(
        Categorical::new(vec![]).unwrap_err().code(),
        ErrorCode::InvalidProbability
    );
    assert_eq!(
        Categorical::uniform(0).unwrap_err().code(),
        ErrorCode::InvalidCount
    );
}

#[test]
fn test_error_display_formatting() {
    // Test that error display includes all expected information
    let err = Normal::new(0.0, -2.5).unwrap_err();
    let display = format!("{}", err);

    // Should include error code
    assert!(display.contains("[101]")); // InvalidVariance

    // Should include distribution name
    assert!(display.contains("Normal"));

    // Should include reason
    assert!(display.contains("must be positive and finite"));

    // Should include parameter value context
    assert!(display.contains("sigma=-2.5"));

    // Should include expected value context
    assert!(display.contains("expected=> 0.0 and finite"));
}

#[test]
fn test_error_descriptions() {
    // Test that error codes have meaningful descriptions
    assert_eq!(
        ErrorCode::InvalidMean.description(),
        "Distribution mean parameter is invalid"
    );
    assert_eq!(
        ErrorCode::InvalidVariance.description(),
        "Distribution variance/scale parameter is invalid"
    );
    assert_eq!(
        ErrorCode::TraceAddressNotFound.description(),
        "Address not found in trace"
    );
    assert_eq!(
        ErrorCode::TypeMismatch.description(),
        "Type mismatch in trace value"
    );

    // Test category descriptions
    assert_eq!(
        ErrorCode::InvalidMean.category(),
        ErrorCategory::DistributionValidation
    );
    assert_eq!(
        ErrorCode::NumericalOverflow.category(),
        ErrorCategory::NumericalComputation
    );
    assert_eq!(
        ErrorCode::TraceAddressNotFound.category(),
        ErrorCategory::TraceManipulation
    );
}
