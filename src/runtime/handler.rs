use crate::core::address::Address;
use crate::core::distribution::Normal;
use crate::core::model::Model;
use crate::runtime::trace::Trace;
pub trait Handler { fn on_sample(&mut self, addr:&Address, dist:&Normal) -> f64; fn on_observe(&mut self, addr:&Address, dist:&Normal, value:f64); fn on_factor(&mut self, logw:f64); fn finish(self)->Trace where Self:Sized; }
pub fn run<A>(mut h: impl Handler, m: Model<A>)->(A, Trace){ fn go<A>(h:&mut impl Handler, m:Model<A>)->A{ match m{ Model::Pure(a)=>a, Model::SampleF{addr,dist,k}=>{ let x=h.on_sample(&addr,&dist); go(h,k(x)) }, Model::ObserveF{addr,dist,value,k}=>{ h.on_observe(&addr,&dist,value); go(h,k(())) }, Model::FactorF{logw,k}=>{ h.on_factor(logw); go(h,k(())) } } } let a=go(&mut h,m); let t=h.finish(); (a,t) }
