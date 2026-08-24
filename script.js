const $=s=>document.querySelector(s);
const fileInput=$("#fileInput"), dropZone=$("#dropZone"), chooseBtn=$("#chooseBtn"), statusEl=$("#status");
const result=$("#result"), testImage=$("#testImage"), canvas=$("#landmarkCanvas"), ctx=canvas.getContext("2d");
const scoreEl=$("#score"), commentEl=$("#comment"), exposureEl=$("#exposureRate"), exposureBar=$("#exposureBar");
const meter=$("#meter"), note=$("#measureNote"), toggle=$("#landmarkToggle"), again=$("#againBtn");

let targetLandmarks=null, targetExposure=25, currentObjectUrl=null, busy=false;
const comments=[
  [40,"いまいちやな"],[50,"今日は仕方なし、これでいいや"],[60,"まあ、いけるべ"],[70,"ふう"],[80,"ごっつええなぁ"],[90,"んっ…"],
  [95,"この世界が丸い球体で出来ていると初めて気づいた人はこの銀河でさえも球体で構成された円で出来ていることに気づいていたのだろうか。この世はこんなにも円で構成されているのに円だけが数学ではっきりした他を持たない。自分自身が原子で構成されているのにもかかわらず自分自身を求めることはできないのだ。"]
];

function setStatus(t){statusEl.textContent=t}
function commentFor(s){let v="測定完了。"; for(const [min,c] of comments) if(s>=min)v=c; return v}
function strictScore(x){
  // 厳しめ：旧70→約40、旧90→約97。0/100は維持。
  x=Math.max(0,Math.min(100,x));
  if(x<=70) return Math.round(x*(40/70));
  if(x<=90) return Math.round(40+(x-70)*(57/20));
  return Math.round(97+(x-90)*(3/10));
}
function bgFor(s){
  const h=215-(Math.max(0,Math.min(100,s))*1.75);
  document.body.style.background=`radial-gradient(circle at 50% -10%,hsl(${h},70%,38%) 0,hsl(${h-18},58%,20%) 42%,#170b18 100%)`;
}
function drawLandmarks(lm){
  const w=canvas.width=testImage.clientWidth*devicePixelRatio, h=canvas.height=testImage.clientHeight*devicePixelRatio;
  canvas.style.width=testImage.clientWidth+"px"; canvas.style.height=testImage.clientHeight+"px";
  ctx.clearRect(0,0,w,h); if(!toggle.checked||!lm)return;
  ctx.fillStyle="#ffd76b";
  for(let i=0;i<lm.length;i+=2){ctx.beginPath();ctx.arc(lm[i].x*w, lm[i].y*h, 1.6*devicePixelRatio,0,Math.PI*2);ctx.fill()}
}
function loadImage(img,src){return new Promise((res,rej)=>{img.onload=()=>res(img);img.onerror=rej;img.src=src})}

async function getFaceLandmarks(img){
  // MediaPipe Tasks is optional. If CDN/model is unavailable, return null rather than hanging.
  try{
    const mod=await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/+esm");
    const vision=await mod.FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm");
    const detector=await mod.FaceLandmarker.createFromOptions(vision,{
      baseOptions:{modelAssetPath:"https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",delegate:"GPU"},
      runningMode:"IMAGE",numFaces:2
    });
    const r=detector.detect(img);
    detector.close?.();
    if(!r.faceLandmarks?.length)return null;
    return r.faceLandmarks[0];
  }catch(e){console.warn("Face AI unavailable:",e);return null}
}
function normalizedSimilarity(a,b){
  if(!a||!b||a.length!==b.length)return null;
  const ids=[33,263,1,61,291,152,10,234,454];
  const get=(arr,i)=>arr[i]||arr[0];
  const le=get(a,33),re=get(a,263), le2=get(b,33),re2=get(b,263);
  const ax=(le.x+re.x)/2, ay=(le.y+re.y)/2, bx=(le2.x+re2.x)/2, by=(le2.y+re2.y)/2;
  const scaleA=Math.hypot(le.x-re.x,le.y-re.y)||1, scaleB=Math.hypot(le2.x-re2.x,le2.y-re2.y)||1;
  let sum=0,n=0;
  for(const i of ids){const p=get(a,i),q=get(b,i);const dx=(p.x-ax)/scaleA-(q.x-bx)/scaleB;const dy=(p.y-ay)/scaleA-(q.y-by)/scaleB;sum+=Math.hypot(dx,dy);n++}
  const d=sum/n;
  return Math.max(0,Math.min(100,100*Math.exp(-d*4.4)));
}
function estimateExposure(img){
  // Conservative, non-sensitive visual-area proxy. It never infers hidden body parts.
  const c=document.createElement("canvas"), w=120,h=Math.max(80,Math.round(120*img.naturalHeight/img.naturalWidth));
  c.width=w;c.height=h;const x=c.getContext("2d");x.drawImage(img,0,0,w,h);
  const d=x.getImageData(0,0,w,h).data; let skin=0,total=0;
  for(let i=0;i<d.length;i+=4){const r=d[i],g=d[i+1],b=d[i+2]; const mx=Math.max(r,g,b),mn=Math.min(r,g,b);
    const skinish=r>65&&g>35&&b>20&&r>g*1.12&&g>b*1.08&&(mx-mn)>20;
    if(skinish)skin++;total++;
  }
  // Normalize to a game-friendly 0..100 visual exposure rate.
  return Math.round(Math.max(0,Math.min(100,(skin/total)*260)));
}
async function setTarget(){
  try{
    const img=new Image(); img.src="target.png"; await img.decode();
    targetLandmarks=await getFaceLandmarks(img);
    // Requested reference exposure: 25%.
  }catch(e){targetLandmarks=null}
}
async function measure(file){
  if(busy)return; busy=true;
  setStatus("測定器、計測開始。");
  note.textContent="顔の特徴を解析中……";
  try{
    if(!file||!/^image\/(jpeg|png|webp)$/.test(file.type)){throw new Error("format")}
    if(currentObjectUrl)URL.revokeObjectURL(currentObjectUrl);
    currentObjectUrl=URL.createObjectURL(file);
    await loadImage(testImage,currentObjectUrl);
    result.hidden=false;
    const lm=await Promise.race([getFaceLandmarks(testImage),new Promise(r=>setTimeout(()=>r(null),12000))]);
    window.lastLandmarks=lm; drawLandmarks(lm);
    const exp=estimateExposure(testImage);
    exposureEl.textContent=exp+"%"; exposureBar.style.width=exp+"%";
    let faceSim=lm&&targetLandmarks?normalizedSimilarity(lm,targetLandmarks):72;
    // Exposure is part of the overall score, not a separate bonus.
    let raw=faceSim==null?Math.max(35,72*0.75+exp*0.25):(faceSim*0.75+exp*0.25);
    // Keep reference-like exposure from dominating while still visibly affecting the game.
    raw=Math.max(0,Math.min(100,raw));
    const s=strictScore(raw);
    scoreEl.textContent=s; commentEl.textContent=commentFor(s);
    bgFor(s); meter.querySelector(".rod").style.transform=`scaleY(${Math.max(.05,s/100)})`;
    meter.classList.toggle("hot",s>=70);
    note.textContent=lm?"測定完了。謎の測定器が結論を出しました。":"測定完了。AIの一部が休憩中ですが、ゲームは続行します。";
    setStatus("測定完了！");
  }catch(e){
    console.warn(e);
    // Never strand the UI in a retry state.
    result.hidden=false; scoreEl.textContent="--"; commentEl.textContent="写真をもう一度投入してみて。";
    note.textContent="測定器は待機中です。";
    setStatus("測定器、待機中。");
  }finally{busy=false}
}
chooseBtn.onclick=()=>fileInput.click();
dropZone.addEventListener("click",e=>{if(e.target!==chooseBtn)fileInput.click()});
fileInput.addEventListener("change",()=>fileInput.files[0]&&measure(fileInput.files[0]));
["dragenter","dragover"].forEach(e=>dropZone.addEventListener(e,x=>{x.preventDefault();dropZone.classList.add("over")}));
["dragleave","drop"].forEach(e=>dropZone.addEventListener(e,x=>{x.preventDefault();dropZone.classList.remove("over")}));
dropZone.addEventListener("drop",e=>{const f=e.dataTransfer.files[0];if(f)measure(f)});
toggle.addEventListener("change",()=>drawLandmarks(window.lastLandmarks||null));
again.onclick=()=>{result.hidden=true;fileInput.value="";setStatus("測定器、待機中。");window.scrollTo({top:0,behavior:"smooth"})};

window.addEventListener("resize",()=>{if(window.lastLandmarks)drawLandmarks(window.lastLandmarks)});
setTarget();
