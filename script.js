const $=s=>document.querySelector(s);
const fileInput=$("#fileInput"),dropZone=$("#dropZone"),chooseBtn=$("#chooseBtn"),statusEl=$("#status");
const result=$("#result"),img=$("#testImage"),canvas=$("#landmarkCanvas"),ctx=canvas.getContext("2d");
const scoreEl=$("#score"),commentEl=$("#comment"),exposureEl=$("#exposureRate"),exposureBar=$("#exposureBar");
const meter=$("#meter"),note=$("#measureNote"),toggle=$("#landmarkToggle"),again=$("#againBtn");

let targetLM=null,targetSignature=null,busy=false,lastLM=null,objectURL=null;

const comments=[
 [40,"いまいちやな"],
 [50,"今日は仕方なし、これでいいや"],
 [60,"まあ、いけるべ"],
 [70,"ふう"],
 [80,"ごっつええなぁ"],
 [90,"んっ…"],
 [95,"この世界が丸い球体で出来ていると初めて気づいた人はこの銀河でさえも球体で構成された円で出来ていることに気づいていたのだろうか。この世はこんなにも円で構成されているのに円だけが数学ではっきりした他を持たない。自分自身が原子で構成されているのにもかかわらず自分自身を求めることはできないのだ。"]
];
function commentFor(s){let c="測定完了。";for(const [n,t] of comments)if(s>=n)c=t;return c}
function strictScore(x){x=Math.max(0,Math.min(100,x));if(x<=70)return Math.round(x*40/70);if(x<=90)return Math.round(40+(x-70)*57/20);return Math.round(97+(x-90)*3/10)}
function bg(s){const h=220-s*1.7;document.body.style.background=`radial-gradient(circle at 50% -10%,hsl(${h},70%,40%) 0,hsl(${h-18},60%,20%) 40%,#1a0d14 100%)`}
function drawLM(lm){canvas.width=img.clientWidth*devicePixelRatio;canvas.height=img.clientHeight*devicePixelRatio;canvas.style.width=img.clientWidth+"px";canvas.style.height=img.clientHeight+"px";ctx.clearRect(0,0,canvas.width,canvas.height);if(!toggle.checked||!lm)return;ctx.fillStyle="#ffe080";for(const p of lm){ctx.beginPath();ctx.arc(p.x*canvas.width,p.y*canvas.height,1.5*devicePixelRatio,0,Math.PI*2);ctx.fill()}}
async function faceAI(image){
 try{
  const m=await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.24/+esm");
  const fs=await m.FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.24/wasm");
  const detector=await m.FaceLandmarker.createFromOptions(fs,{baseOptions:{modelAssetPath:"https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",delegate:"GPU"},runningMode:"IMAGE",numFaces:2});
  const r=detector.detect(image);detector.close?.();return r.faceLandmarks||[];
 }catch(e){console.warn("Face AI unavailable",e);return []}
}
function centerScale(lm){
 const L=lm[33]||lm[1],R=lm[263]||lm[1],N=lm[1]||lm[0];
 const cx=(L.x+R.x)/2,cy=(L.y+R.y)/2;
 const dx=R.x-L.x,dy=R.y-L.y,scale=Math.hypot(dx,dy)||1;
 const ang=Math.atan2(dy,dx),co=Math.cos(-ang),si=Math.sin(-ang);
 return lm.map(p=>{const x=(p.x-cx)/scale,y=(p.y-cy)/scale;return {x:x*co-y*si,y:x*si+y*co,z:(p.z||0)/scale}})
}
function signature(lm){
 const n=centerScale(lm), ids=[10,152,234,454,33,263,159,145,386,374,1,4,61,291,13,14,50,280,94,334,199];
 return ids.map(i=>{const p=n[i]||n[0];return [p.x,p.y,p.z]})
}
function similarity(a,b){
 if(!a||!b||a.length!==b.length)return null;
 let d=0;for(let i=0;i<a.length;i++){const dx=a[i][0]-b[i][0],dy=a[i][1]-b[i][1],dz=a[i][2]-b[i][2];d+=Math.sqrt(dx*dx+dy*dy+dz*dz)}
 d/=a.length;
 // Same normalized landmark shape maps very close to 100.
 return Math.max(0,Math.min(100,100*Math.exp(-d*6.0)))
}
function exposure(img){
 // Visual skin-area proxy, excluding the central face area so the face itself does not inflate exposure.
 const c=document.createElement("canvas"),w=140,h=Math.max(100,Math.round(140*img.naturalHeight/img.naturalWidth)),x=c.getContext("2d");c.width=w;c.height=h;x.drawImage(img,0,0,w,h);
 const d=x.getImageData(0,0,w,h).data;let s=0,t=0;
 for(let y=0;y<h;y++)for(let xx=0;xx<w;xx++){const nx=(xx/w-.5)*2,ny=(y/h-.43)*2;if(nx*nx/0.18+ny*ny/0.28<1)continue;const k=(y*w+xx)*4,r=d[k],g=d[k+1],b=d[k+2];if(r>65&&r>g*1.1&&g>b*1.05&&r-g>12)s++;t++}
 return Math.round(Math.max(0,Math.min(100,s/t*280)))
}
async function loadTarget(){
 try{
  const t=new Image();t.src="target.png";await t.decode();
  const arr=await Promise.race([faceAI(t),new Promise(r=>setTimeout(()=>r([]),12000))]);
  if(arr.length===1){targetLM=arr[0];targetSignature=signature(targetLM)}
 }catch(e){console.warn("Target load failed",e)}
}
async function measure(file){
 if(busy)return;busy=true;statusEl.textContent="測定器、計測開始。";note.textContent="顔の特徴を解析中……";
 try{
  if(!file||!/^image\/(jpeg|png|webp)$/.test(file.type))throw Error("format");
  if(objectURL)URL.revokeObjectURL(objectURL);objectURL=URL.createObjectURL(file);
  img.onload=()=>drawLM(lastLM);img.src=objectURL;await img.decode();result.hidden=false;
  const arr=await Promise.race([faceAI(img),new Promise(r=>setTimeout(()=>r([]),12000))]);
  if(arr.length>1){note.textContent="顔が複数見つかりました。1人で投入してね。";statusEl.textContent="測定器、待機中。";busy=false;return}
  lastLM=arr[0]||null;drawLM(lastLM);
  const ex=exposure(img);exposureEl.textContent=ex+"%";exposureBar.style.width=ex+"%";
  let face=targetSignature&&lastLM?similarity(signature(lastLM),targetSignature):null;
  // Robust fallback: if AI is unavailable, do not pretend the result is an exact identity match.
  if(face==null)face=55;
  // Exposure is integrated into the 100-point total; reference exposure is 25%.
  const expContribution=Math.max(0,Math.min(100,100-Math.abs(ex-25)*1.15));
  const raw=face*.78+expContribution*.22;
  const finalScore=strictScore(raw);
  scoreEl.textContent=finalScore;commentEl.textContent=commentFor(finalScore);
  bg(finalScore);meter.querySelector(".rod").style.transform=`scaleY(${Math.max(.05,finalScore/100)})`;meter.classList.toggle("hot",finalScore>=70);
  note.textContent=targetSignature?"測定完了。顔の特徴を比較しました。":"測定完了。基準AIの準備が不十分なため暫定比較です。";
  statusEl.textContent="測定完了！";
 }catch(e){console.warn(e);result.hidden=false;scoreEl.textContent="--";commentEl.textContent="写真をもう一度投入してみて。";note.textContent="測定器、待機中。";statusEl.textContent="測定器、待機中。"}
 finally{busy=false}
}
chooseBtn.onclick=e=>{e.stopPropagation();fileInput.click()};
dropZone.onclick=e=>{if(e.target!==chooseBtn)fileInput.click()};
fileInput.onchange=()=>fileInput.files[0]&&measure(fileInput.files[0]);
dropZone.ondragover=e=>e.preventDefault();
dropZone.ondrop=e=>{e.preventDefault();const f=e.dataTransfer.files[0];if(f)measure(f)};
toggle.onchange=()=>drawLM(lastLM);
again.onclick=()=>{result.hidden=true;fileInput.value="";statusEl.textContent="測定器、待機中。";window.scrollTo({top:0,behavior:"smooth"})};
window.onresize=()=>drawLM(lastLM);
loadTarget();
