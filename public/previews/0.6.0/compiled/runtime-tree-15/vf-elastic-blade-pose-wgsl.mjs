// One material-space blade map for World contact and View/shadow geometry.
// The scalar bend is a physical mode supplied by the World, not animation time.
export const ELASTIC_BLADE_POSE_WGSL = /* wgsl */`
struct ElasticBladePose { p:vec3<f32>, n:vec3<f32>, dp_dangle:vec3<f32>, dn_dangle:vec3<f32> };
fn elastic_blade_trig(angle:f32)->vec2<f32>{
 let a=angle-6.28318530718*floor((angle+3.14159265359)/6.28318530718);let folded=abs(a)>1.57079632679;let x=select(a,sign(a)*3.14159265359-a,folded);let q=x*x;
 let s=x*(1.0+q*(-0.166666666666667+q*(0.00833333333333333+q*(-0.000198412698412698+q*(0.00000275573192239859+q*(-0.0000000250521083854417+q*0.000000000160590438368216))))));
 let c=(1.0+q*(-0.5+q*(0.0416666666666667+q*(-0.00138888888888889+q*(0.0000248015873015873+q*(-0.000000275573192239859+q*(0.00000000208767569878681-q*0.0000000000114707455977297)))))))*select(1.0,-1.0,folded);
 return vec2<f32>(s,c);
}
fn elastic_blade_rotate(p:vec3<f32>,axis:vec3<f32>,angle:f32)->vec3<f32>{
 let trig=elastic_blade_trig(angle);return p*trig.y+cross(axis,p)*trig.x+axis*dot(axis,p)*(1.0-trig.y);
}
// sinc, cosc and their derivatives have removable singularities at zero.
fn elastic_blade_coefficients(a:f32)->vec4<f32>{
 let q=a*a;
 if(abs(a)<0.01){return vec4<f32>(1.0-q/6.0+q*q/120.0,a*(0.5-q/24.0+q*q/720.0),a*(-1.0/3.0+q/30.0-q*q/840.0),0.5-q/8.0+q*q/144.0);}
 let trig=elastic_blade_trig(a);let s=trig.x;let c=trig.y;
 return vec4<f32>(s/a,(1.0-c)/a,(a*c-s)/q,(a*s-(1.0-c))/q);
}
fn elastic_blade_pose(rest:vec3<f32>,normal:vec3<f32>,anchor:vec3<f32>,hinge:vec3<f32>,blade_length:f32,bend:f32,translation:vec3<f32>)->ElasticBladePose{
 let offset=rest-anchor;let axis=normalize(cross(normal,hinge));
 let longitudinal=dot(offset,axis);let width=dot(offset,hinge);let camber=dot(offset,normal);
 let along=clamp(longitudinal/max(blade_length,0.02),0.0,1.0);let a=bend*along;let coeff=elastic_blade_coefficients(a);
 let n=elastic_blade_rotate(normal,hinge,a);let dn=cross(hinge,n)*along;
 let p=anchor+longitudinal*(axis*coeff.x+normal*coeff.y)+hinge*width+n*camber+translation;
 let dp=longitudinal*(axis*coeff.z+normal*coeff.w)*along+dn*camber;
 return ElasticBladePose(p,n,dp,dn);
}
// For a linearly varying bend, this bounds every curved vertex path over an
// interval, not just its endpoint chord. Add the translation speed separately.
fn elastic_blade_bend_speed_bound(rest:vec3<f32>,normal:vec3<f32>,anchor:vec3<f32>,hinge:vec3<f32>,blade_length:f32,delta_bend:f32)->f32{
 let offset=rest-anchor;let longitudinal=dot(offset,normalize(cross(normal,hinge)));
 let along=clamp(longitudinal/max(blade_length,0.02),0.0,1.0);
 return along*(0.5*abs(longitudinal)+abs(dot(offset,normal)))*abs(delta_bend);
}
`;
