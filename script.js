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
function strictScore(x){
 x=Math.max(0,Math.min(100,x));

 if(x<=70)
  return Math.round(x*40/70);

 if(x<=90)
  return Math.round(
   40+(x-70)*57/20
  );

 return Math.round(
   97+(x-90)*3/10
  );
}
function bg(s){const h=220-s*1.7;document.body.style.background=`radial-gradient(circle at 50% -10%,hsl(${h},70%,40%) 0,hsl(${h-18},60%,20%) 40%,#1a0d14 100%)`}
function drawLM(lm){
 canvas.width=img.clientWidth*devicePixelRatio;
 canvas.height=img.clientHeight*devicePixelRatio;
 canvas.style.width=img.clientWidth+"px";
 canvas.style.height=img.clientHeight+"px";

 ctx.clearRect(0,0,canvas.width,canvas.height);

 if(!toggle.checked||!lm)return;

 // imgの実際の表示領域（object-fit: contain）を計算
 const iw=img.naturalWidth;
 const ih=img.naturalHeight;
 const cw=img.clientWidth;
 const ch=img.clientHeight;

 const scale=Math.min(cw/iw,ch/ih);

 const dw=iw*scale;
 const dh=ih*scale;

 const offsetX=(cw-dw)/2;
 const offsetY=(ch-dh)/2;

 // 全顔ランドマーク
 ctx.fillStyle="#ffe080";

 for(const p of lm){
  const x=(offsetX+p.x*dw)*devicePixelRatio;
  const y=(offsetY+p.y*dh)*devicePixelRatio;

  ctx.beginPath();
  ctx.arc(
   x,
   y,
   1.5*devicePixelRatio,
   0,
   Math.PI*2
  );
  ctx.fill();
 }

 // 左目
 const leftEye=[
  33,133,159,145,160,161,158,144,153,154,155
 ];

 // 右目
 const rightEye=[
  362,263,386,374,387,388,385,373,380,381,382
 ];

 // 目を見やすく表示
 ctx.fillStyle="#00ffff";

 for(const i of [...leftEye,...rightEye]){
  const p=lm[i];
  if(!p)continue;

  const x=(offsetX+p.x*dw)*devicePixelRatio;
  const y=(offsetY+p.y*dh)*devicePixelRatio;

  ctx.beginPath();
  ctx.arc(
   x,
   y,
   4*devicePixelRatio,
   0,
   Math.PI*2
  );
  ctx.fill();
 }
}
async function faceAI(image){
 try{
  const m=await import(
   "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/+esm"
  );

  const fs=await m.FilesetResolver.forVisionTasks(
   "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm"
  );

  const detector=await m.FaceLandmarker.createFromOptions(
   fs,
   {
    baseOptions:{
     modelAssetPath:
      "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
     delegate:"GPU"
    },
    runningMode:"IMAGE",
    numFaces:2
   }
  );

  const r=detector.detect(image);

  detector.close?.();

  return r.faceLandmarks||[];

 }catch(e){
  console.warn("Face AI unavailable",e);
  return [];
 }
}
function centerScale(lm){
 const L=lm[33]||lm[1];
 const R=lm[263]||lm[1];

 const cx=(L.x+R.x)/2;
 const cy=(L.y+R.y)/2;

 const dx=R.x-L.x;
 const dy=R.y-L.y;

 const scale=Math.hypot(dx,dy)||1;

 const ang=Math.atan2(dy,dx);
 const co=Math.cos(-ang);
 const si=Math.sin(-ang);

 return lm.map(p=>{
  const x=(p.x-cx)/scale;
  const y=(p.y-cy)/scale;

  return {
   x:x*co-y*si,
   y:x*si+y*co,
   z:(p.z||0)/scale
  };
 });
}


function signature(lm){
 const n=centerScale(lm);

 /*
  表情や撮影条件の影響を受けにくい
  顔の骨格・パーツ配置を中心に比較する
 */

 const ids=[
  // 輪郭
  10,152,234,454,
  127,356,93,323,
  132,361,58,288,

  // 眉～目
  33,133,159,145,
  362,263,386,374,

  // 鼻
  1,2,4,5,6,
  19,94,168,

  // 口
  61,291,13,14,

  // その他の顔形状
  50,280,199,175,18
 ];

 return ids.map(i=>{
  const p=n[i]||n[0];
  return [p.x,p.y,p.z];
 });
}


function shapeSimilarity(a,b){
 if(!a||!b||a.length!==b.length)return 0;

 let d=0;

 for(let i=0;i<a.length;i++){
  const dx=a[i][0]-b[i][0];
  const dy=a[i][1]-b[i][1];

  /*
   Zは撮影角度の影響を受けやすいので
   XYを強く評価する
  */
  const dz=(a[i][2]-b[i][2])*0.35;

  d+=Math.sqrt(
   dx*dx+
   dy*dy+
   dz*dz
  );
 }

 d/=a.length;

 /*
  小さな差は同一人物として許容し、
  大きな差は急激に減点する
  */
 return Math.max(
  0,
  Math.min(
   100,
   100*Math.exp(-d*3.8)
  )
 );
}


function similarity(a,b){
 if(!a||!b)return null;

 const raw=shapeSimilarity(a,b);

 /*
  同一人物の別写真では
  多少の表情・角度変化を許容する。

  一方、別人の場合は
  一定以上の形状差を大きく減点する。
 */

 let score;

 if(raw>=75){
  score=75+(raw-75)*1.0;
 }
 else if(raw>=50){
  score=45+(raw-50)*1.2;
 }
 else{
  score=raw*0.72;
 }

 return Math.max(
  0,
  Math.min(100,score)
 );
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

if(face==null){
  note.textContent="基準画像の解析に失敗しました。";
  statusEl.textContent="測定器、待機中。";
  busy=false;
  return;
}
const expContribution=Math.max(
  0,
  Math.min(100,100-Math.abs(ex-25)*1.15)
);

const raw=face*0.90+expContribution*0.10;
const finalScore=Math.round(raw);
scoreEl.textContent=finalScore;commentEl.textContent=commentFor(finalScore);
bg(finalScore);meter.querySelector(".rod").style.transform=`scaleY(${Math.max(.05,finalScore/100)})`;meter.classList.toggle("hot",finalScore>=70);
note.textContent="測定完了。顔の特徴を比較しました。";
statusEl.textContent="測定完了！";
 }catch(e){console.warn(e);result.hidden=false;scoreEl.textContent="--";commentEl.textContent="写真をもう一度投入してみて。";note.textContent="測定器、待機中。";statusEl.textContent="測定器、待機中。"}
 finally{busy=false}
}
chooseBtn.onclick=e=>{
 e.stopPropagation();
 fileInput.click();
};
fileInput.onchange=()=>{
 const f=fileInput.files[0];
 if(f)measure(f);
};

dropZone.ondragover=e=>e.preventDefault();

dropZone.ondrop=e=>{
 e.preventDefault();
 const f=e.dataTransfer.files[0];
 if(f)measure(f);
};

toggle.onchange=()=>drawLM(lastLM);

again.onclick=()=>{
 result.hidden=true;
 fileInput.value="";
 statusEl.textContent="測定器、待機中。";
 window.scrollTo({top:0,behavior:"smooth"});
};

window.onresize=()=>drawLM(lastLM);

loadTarget();
