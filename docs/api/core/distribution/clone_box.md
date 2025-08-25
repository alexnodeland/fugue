# `clone_box` function

Clone this distribution into a boxed trait object.

This method is required for the trait to be object-safe, allowing distributions to be stored as `Box<dyn Distribution<T>>`.
