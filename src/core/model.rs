use crate::core::address::Address;
use crate::core::distribution::{LogF64, Normal};
pub enum Model<A> {
  Pure(A),
  SampleF { addr: Address, dist: Normal, k: Box<dyn FnOnce(f64)->Model<A> + Send> },
  ObserveF { addr: Address, dist: Normal, value: f64, k: Box<dyn FnOnce(()) -> Model<A> + Send> },
  FactorF { logw: LogF64, k: Box<dyn FnOnce(()) -> Model<A> + Send> },
}
pub fn pure<A>(a:A)->Model<A>{Model::Pure(a)}
pub fn sample(addr: Address, dist: Normal)->Model<f64>{ Model::SampleF{addr, dist, k: Box::new(pure)} }
pub fn observe(addr: Address, dist: Normal, value: f64)->Model<()> { Model::ObserveF{addr, dist, value, k: Box::new(pure)} }
pub fn factor(logw: LogF64)->Model<()> { Model::FactorF{logw, k: Box::new(pure)} }
pub trait ModelExt<A>: Sized { fn bind<B>(self, k: impl FnOnce(A)->Model<B> + Send + 'static)->Model<B>; fn map<B>(self, f: impl FnOnce(A)->B + Send + 'static)->Model<B>{ self.bind(|a| pure(f(a))) } }
impl<A> ModelExt<A> for Model<A> { fn bind<B>(self, k: impl FnOnce(A)->Model<B> + Send + 'static)->Model<B>{ match self {
  Model::Pure(a)=>k(a),
  Model::SampleF{addr,dist,k:k1}=>Model::SampleF{addr,dist,k:Box::new(move|x|k1(x).bind(k))},
  Model::ObserveF{addr,dist,value,k:k1}=>Model::ObserveF{addr,dist,value,k:Box::new(move|u|k1(u).bind(k))},
  Model::FactorF{logw,k:k1}=>Model::FactorF{logw,k:Box::new(move|u|k1(u).bind(k))}, }}}
