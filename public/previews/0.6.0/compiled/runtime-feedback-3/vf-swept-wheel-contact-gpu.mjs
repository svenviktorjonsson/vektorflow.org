// Shared bounded analytic contact truth for the liquid/granular adapters.
export const SWEPT_WHEEL_CONTACT_WGSL = /* wgsl */`
struct SweptHit { time:f32, normal:vec2<f32>, kind:u32 };
struct SweptMotion { position:vec2<f32>, velocity:vec2<f32> };
fn no_swept_hit()->SweptHit{return SweptHit(1.00001,vec2<f32>(0.0),0u);}
fn ray_hit(best:SweptHit,t:f32,n:vec2<f32>,d:vec2<f32>,kind:u32)->SweptHit{
  if(t>=-1.0e-6&&t<=1.0&&t<best.time&&dot(d,n)<-1.0e-9){return SweptHit(max(0.0,t),n,kind);}return best;
}
fn capsule_ray(best_input:SweptHit,p:vec2<f32>,d:vec2<f32>,a:vec2<f32>,b:vec2<f32>,r:f32)->SweptHit{
  var best=best_input;let edge=b-a;let length_edge=length(edge);let axis=edge/max(length_edge,1.0e-12);let n=vec2<f32>(-axis.y,axis.x);
  for(var side=-1.0;side<=1.0;side+=2.0){let normal=n*side;let speed=dot(d,normal);if(speed< -1.0e-9){let t=(r-dot(p-a,normal))/speed;let along=dot(p+d*t-a,axis);if(along>=0.0&&along<=length_edge){best=ray_hit(best,t,normal,d,2u);}}}
  let aa=dot(d,d);if(aa>1.0e-18){for(var end=0u;end<2u;end++){let c=select(a,b,end==1u);let v=p-c;let bb=dot(v,d);let cc=dot(v,v)-r*r;let disc=bb*bb-aa*cc;
    if(disc>=0.0){let t=(-bb-sqrt(disc))/aa;let point=p+d*t;let along=dot(point-a,axis);if((end==0u&&along<=0.0)||(end==1u&&along>=length_edge)){best=ray_hit(best,t,normalize(point-c),d,2u);}}
  }}return best;
}
fn wheel_ray(p:vec2<f32>,d:vec2<f32>,center:vec2<f32>,limit:f32,contact_radius:f32,angle:f32)->SweptHit{
  var best=no_swept_hit();for(var segment=0u;segment < 7u;segment++){let local=baffle(segment);best=capsule_ray(best,p,d,center+rotate_local(local.xy,angle),center+rotate_local(local.zw,angle),contact_radius);}
  let v=p-center;let aa=dot(d,d);let bb=dot(v,d);let cc=dot(v,v)-limit*limit;let disc=bb*bb-aa*cc;
  if(aa>1.0e-18&&disc>=0.0){let t=(-bb+sqrt(disc))/aa;best=ray_hit(best,t,-normalize(v+d*t),d,1u);}return best;
}
fn arc_hit(best:SweptHit,point:vec2<f32>,normal:vec2<f32>,origin:vec2<f32>,delta:f32,center:vec2<f32>)->SweptHit{
  let v=origin-center;let q=point-center;let direction=select(-1.0,1.0,delta>0.0);
  if(dot(vec2<f32>(-q.y,q.x)*direction,normal)>=-1.0e-9){return best;}
  var travel=direction*atan2(v.x*q.y-v.y*q.x,dot(v,q));if(abs(travel)<1.0e-6){travel=0.0;}if(travel<0.0){travel+=6.28318530718;}
  let t=travel/max(abs(delta),1.0e-12);if(t<=1.0&&t<best.time){return SweptHit(t,normal,2u);}return best;
}
fn wheel_arc(origin:vec2<f32>,delta:f32,center:vec2<f32>,r:f32,angle:f32)->SweptHit{
  var best=no_swept_hit();let radius=length(origin-center);if(radius<1.0e-9||abs(delta)<1.0e-12){return best;}
  for(var segment=0u;segment < 7u;segment++){
    let local=baffle(segment);let a=rotate_local(local.xy,angle);let b=rotate_local(local.zw,angle);let edge=b-a;let aa=dot(edge,edge);let axis=edge/max(length(edge),1.0e-12);let normal=vec2<f32>(-axis.y,axis.x);
    for(var side=-1.0;side<=1.0;side+=2.0){let start=a+normal*side*r;let bb=dot(start,edge);let cc=dot(start,start)-radius*radius;let disc=bb*bb-aa*cc;
      if(aa>1.0e-18&&disc>=0.0){for(var root=-1.0;root<=1.0;root+=2.0){let t=(-bb+root*sqrt(disc))/aa;if(t>=0.0&&t<=1.0){let point=start+edge*t;best=arc_hit(best,point+center,normal*side,origin,delta,center);}}}
    }
    for(var end=0u;end<2u;end++){let cap=select(a,b,end==1u);let distance=length(cap);if(distance>1.0e-9){let cosine=(radius*radius+distance*distance-r*r)/(2.0*radius*distance);
      if(abs(cosine)<=1.0){for(var root=-1.0;root<=1.0;root+=2.0){let phi=atan2(cap.y,cap.x)+root*acos(cosine);let point=vec2<f32>(cos(phi),sin(phi))*radius;let along=dot(point-a,axis);
        if((end==0u&&along<=0.0)||(end==1u&&along>=length(edge))){best=arc_hit(best,point+center,normalize(point-cap),origin,delta,center);}
      }}
    }}
  }return best;
}
fn swept_wheel_motion(origin:vec2<f32>,destination:vec2<f32>,velocity:vec2<f32>,center:vec2<f32>,limit:f32,r:f32,angle:f32)->SweptMotion{
  var p=origin;var d=destination-origin;var v=velocity;
  for(var contact=0u;contact<8u;contact++){
    if(dot(d,d)<1.0e-18){return SweptMotion(p,v);}let hit=wheel_ray(p,d,center,limit,r,angle);
    if(hit.kind==0u){return SweptMotion(p+d,v);}p+=d*hit.time;d*=1.0-hit.time;
    v-=hit.normal*min(0.0,dot(v,hit.normal));d-=hit.normal*min(0.0,dot(d,hit.normal));
    if(hit.kind==1u){
      let outward=normalize(p-center);let tangent=vec2<f32>(-outward.y,outward.x);let arc=dot(d,tangent)/limit;
      let next=wheel_arc(p,arc,center,r,angle);let fraction=min(1.0,next.time);let rotation=arc*fraction;
      p=center+rotate_local(outward*limit,rotation);v=rotate_local(v,rotation);
      if(next.kind==0u){return SweptMotion(p,v);}
      let final_tangent=rotate_local(tangent,rotation);d=final_tangent*(arc*limit*(1.0-fraction));p+=next.normal*1.0e-7;
      v-=next.normal*min(0.0,dot(v,next.normal));d-=next.normal*min(0.0,dot(d,next.normal));
    }else{p+=hit.normal*1.0e-7;}
  }
  // Fail closed at a singular multi-contact: never consume an unchecked tail.
  return SweptMotion(p,vec2<f32>(0.0));
}
fn swept_wheel_rotation(p:vec2<f32>,v:vec2<f32>,center:vec2<f32>,r:f32,old_angle:f32,delta:f32,elapsed:f32)->SweptMotion{
  let hit=wheel_arc(p,-delta,center,r,old_angle);if(hit.kind==0u){return SweptMotion(p,v);}
  let normal=rotate_local(hit.normal,delta);let position=center+rotate_local(p-center,delta*(1.0-hit.time))+normal*1.0e-7;
  let surface=delta/elapsed*vec2<f32>(-(position-center).y,(position-center).x);
  return SweptMotion(position,v+normal*max(0.0,dot(surface-v,normal)));
}
`;
