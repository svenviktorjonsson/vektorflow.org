// Internal compiler-produced WGSL only. Freeze scalar specializations before
// module creation: older WebKit rejects API overrides unused by an entry point.
// Function bodies, bindings and physical values are not rewritten.
export function specializeWgslOverrides(source,values={}) {
  const visible=source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\r\n]*/g,text=>text.replace(/[^\r\n]/g,' '));
  const pattern=/\boverride\s+([A-Za-z_]\w*)\s*(?::\s*(bool|u32|i32|f32))?\s*=\s*([^;]+);/g;
  const edits=[],known=new Set();let match;
  while((match=pattern.exec(visible))){
    const [,name,declared,initial]=match,defaultValue=initial.trim();known.add(name);
    const type=declared??(/^(true|false)$/.test(defaultValue)?'bool':/^\d+u$/.test(defaultValue)?'u32':/^-?\d+i$/.test(defaultValue)?'i32':/^-?\d+$/.test(defaultValue)?'i32':/^-?(?:\d+\.\d*|\d*\.\d+|\d+[eE][+-]?\d+)(?:f)?$/.test(defaultValue)?'f32':null);
    if(!type)throw new Error(`Unsupported internal WGSL override initializer: ${name}`);
    let literal=source.slice(match.index,match.index+match[0].length).match(/=\s*([^;]+);$/)[1].trim();
    if(Object.hasOwn(values,name)){
      const value=values[name];
      if(type==='bool'){
        if(typeof value!=='boolean'&&value!==0&&value!==1)throw new Error(`Invalid bool specialization: ${name}`);
        literal=value?'true':'false';
      }else{
        if(typeof value!=='number'||!Number.isFinite(value))throw new Error(`Invalid numeric specialization: ${name}`);
        if(type==='u32'||type==='i32'){
          const lower=type==='u32'?0:-2147483648,upper=type==='u32'?4294967295:2147483647;
          if(!Number.isInteger(value)||value<lower||value>upper)throw new Error(`Invalid ${type} specialization: ${name}`);
          literal=String(value)+(type==='u32'?'u':'i');
        }else{
          if(Math.abs(value)>3.4028234663852886e38)throw new Error(`Invalid f32 specialization: ${name}`);
          literal=String(value);if(!/[.eE]/.test(literal))literal+='.0';
        }
      }
    }
    edits.push({start:match.index,end:match.index+match[0].length,text:`const ${name}: ${type} = ${literal};`});
  }
  for(const name of Object.keys(values))if(!known.has(name))throw new Error(`Unknown WGSL specialization: ${name}`);
  for(let i=edits.length-1;i>=0;i--){const edit=edits[i];source=source.slice(0,edit.start)+edit.text+source.slice(edit.end);}
  return source;
}

export function createStaticWgslModuleCache(device,source) {
  const modules=new Map();
  return values=>{
    const key=JSON.stringify(Object.entries(values).sort(([a],[b])=>a.localeCompare(b)));
    if(!modules.has(key))modules.set(key,device.createShaderModule({label:'VKF contact static specialization',code:specializeWgslOverrides(source,values)}));
    return modules.get(key);
  };
}
