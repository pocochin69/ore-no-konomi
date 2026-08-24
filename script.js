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
   * 顔認証用特徴点
   *
   * 表情や撮影条件に比較的強い部分を中心に使用。
   */

  const ids=[
    // 顔の輪郭
    10,152,
    234,454,
    127,356,
    93,323,
    132,361,
    58,288,

    // 目
    33,133,
    159,145,
    362,263,
    386,374,

    // 鼻
    1,2,4,5,6,
    19,94,
    168,

    // 口
    61,291,
    13,14,

    // 顔中央・その他
    50,280,
    199,175,
    18
  ];

  return ids.map(i=>{
    const p=n[i]||n[0];

    return {
      x:p.x,
      y:p.y,
      z:p.z
    };
  });
}


/*
 * 点群そのものではなく、
 * 「点と点の距離の比率」も比較する。
 *
 * これにより
 * ・写真の拡大縮小
 * ・多少の顔の位置ズレ
 * ・カメラとの距離
 * の影響をさらに減らす。
 */
function pairDistances(sig){

  const pairs=[
    [0,1],    // 顔の上下
    [2,3],    // 顔の横幅
    [4,5],    // 左目幅
    [6,7],    // 右目幅
    [8,9],    // 鼻
    [10,11],  // 口幅
    [12,13],  // 顔横
    [14,15],
    [16,17],
    [18,19],
    [20,21],
    [22,23]
  ];

  return pairs.map(([a,b])=>{
    const p=sig[a];
    const q=sig[b];

    return Math.hypot(
      p.x-q.x,
      p.y-q.y
    );
  });
}


/*
 * ランドマーク形状の比較
 */
function shapeSimilarity(a,b){

  if(!a||!b||a.length!==b.length)return 0;

  let distance=0;

  for(let i=0;i<a.length;i++){

    const dx=a[i].x-b[i].x;
    const dy=a[i].y-b[i].y;

    /*
     * Zは顔の向きによる変化が大きいため
     * XYより弱くする。
     */
    const dz=(a[i].z-b[i].z)*0.25;

    distance+=Math.sqrt(
      dx*dx+
      dy*dy+
      dz*dz
    );
  }

  distance/=a.length;


  /*
   * 相対距離の比較
   */
  const da=pairDistances(a);
  const db=pairDistances(b);

  let ratioError=0;

  for(let i=0;i<da.length;i++){

    const aa=da[i];
    const bb=db[i];

    if(aa<0.0001||bb<0.0001)continue;

    ratioError+=Math.abs(
      Math.log(aa/bb)
    );
  }

  ratioError/=da.length;


  /*
   * ランドマーク一致度
   *
   * distanceが小さいほど高得点。
   */
  const pointScore=
    100*Math.exp(-distance*4.0);


  /*
   * 顔パーツ間の比率一致度
   */
  const ratioScore=
    100*Math.exp(-ratioError*2.2);


  /*
   * 形状を重視。
   */
  const score=
    pointScore*0.60+
    ratioScore*0.40;

  return Math.max(
    0,
    Math.min(100,score)
  );
}


/*
 * 最終的な顔類似度
 */
function similarity(a,b){

  if(!a||!b)return null;

  const raw=shapeSimilarity(a,b);

  /*
   * 高い一致度はそのまま評価。
   *
   * 中間域は少し厳しくする。
   * これによって別人が中途半端に
   * 高得点になるのを防ぐ。
   */
  let score;

  if(raw>=90){

    score=
      92+
      (raw-90)*0.8;

  }else if(raw>=75){

    score=
      70+
      (raw-75)*1.47;

  }else if(raw>=55){

    score=
      40+
      (raw-55)*1.5;

  }else{

    score=
      raw*0.70;
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
