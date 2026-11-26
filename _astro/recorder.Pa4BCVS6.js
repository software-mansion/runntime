import{_ as e,a as t}from"./WeightsGate.CK4JtZyC.js";var n=`
  registerProcessor('docs-pcm-capture', class extends AudioWorkletProcessor {
    process(inputs) {
      const channel = inputs[0][0];
      if (channel) this.port.postMessage(channel.slice(0));
      return true;
    }
  });
`;async function r(r=t){let i=await navigator.mediaDevices.getUserMedia({audio:!0}),a;try{a=new AudioContext({sampleRate:r})}catch{a=new AudioContext}let o=URL.createObjectURL(new Blob([n],{type:`text/javascript`}));try{await a.audioWorklet.addModule(o)}finally{URL.revokeObjectURL(o)}let s=[],c=new AudioWorkletNode(a,`docs-pcm-capture`);c.port.onmessage=e=>s.push(e.data),a.createMediaStreamSource(i).connect(c),await a.resume();let l=async()=>{let e=a.sampleRate;for(let e of i.getTracks())e.stop();await a.close();let t=new Float32Array(s.reduce((e,t)=>e+t.length,0)),n=0;for(let e of s)t.set(e,n),n+=e.length;return{samples:t,sampleRate:e}};return{async stop(){let{samples:t,sampleRate:n}=await l();return await e(t,n)},stopRaw:l}}export{r as t};