/* 俺の好み！で一発 // local browser game
   Target image is bundled as target.png. Face analysis uses MediaPipe Face Mesh.
   No uploaded image is sent to a server by this application.
*/
const $ = (s) => document.querySelector(s);
const els = {
  fileInput: $('#fileInput'), dropZone: $('#dropZone'), chooseBtn: $('#chooseBtn'),
  targetImage: $('#targetImage'), targetCanvas: $('#targetLandmarks'), testImage: $('#testImage'), testCanvas: $('#testLandmarks'),
  testPreview: $('#testPreview'), fileStatus: $('#fileStatus'), analysisTitle: $('#analysisTitle'), analysisText: $('#analysisText'),
  progressBar: $('#progressBar'), resultSection: $('#resultSection'), controls: $('#controls'),
  faceScore: $('#faceScore'), exposureScore: $('#exposureScore'), totalScore: $('#totalScore'), verdict: $('#verdict'),
  shapeScore: $('#shapeScore'), eyeScore: $('#eyeScore'), noseScore: $('#noseScore'), mouthScore: $('#mouthScore'),
  meterBar: $('#meterBar'), meterRank: $('#meterRank'), meterCaption: $('#meterCaption'), meterFlare: $('#meterFlare'),
  resetBtn: $('#resetBtn'), landmarkToggle: $('#landmarkToggle'), toast: $('#toast'), sparkLayer: $('#sparkLayer')
};

let faceMesh;
let targetLandmarks = null;
let testLandmarks = null;
let targetImageReady = false;
let testObjectUrl = null;
let showTestLandmarks = true;
let showTargetLandmarks = true;
const TARGET_PATH = 'target.png';

function wait(ms){return new Promise(r=>setTimeout(r,ms));}
function clamp(v,a=0,b=100){return Math.max(a,Math.min(b,v));}
function dist(a,b){return Math.hypot(a.x-b.x,a.y-b.y,a.z-b.z);}
function mid(a,b){return {x:(a.x+b.x)/2,y:(a.y+b.y)/2,z:(a.z+b.z)/2};}
function avg(points){const out={x:0,y:0,z:0}; points.forEach(p=>{out.x+=p.x;out.y+=p.y;out.z+=p.z||0}); return {x:out.x/points.length,y:out.y/points.length,z:out.z/points.length};}

// MediaPipe's standard Face Mesh landmark indices.
const IDX = {
  leftEyeOuter:33,rightEyeOuter:263,leftEyeInner:133,rightEyeInner:362,
  noseTip:1,noseLeft:129,noseRight:358,mouthL:61,mouthR:291,mouthTop:13,mouthBottom:14,
  chin:152,forehead:10,leftCheek:234,rightCheek:454,leftBrow:105,rightBrow:334,
  leftUpper:159,leftLower:145,rightUpper:386,rightLower:374
};
const contourIdx = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109];
const eyeLIdx=[33,133,159,145,160,144,158,153];
const eyeRIdx=[362,263,386,374,387,373,385,380];
const noseIdx=[1,2,4,5,6,19,94,97,98,168,197,195,236,3];
const mouthIdx=[61,291,13,14,78,308,80,310,81,311,82,312,87,317,95,324,178,402];
const browIdx=[70,63,105,66,107,336,296,334,293,300];

function normalizedPoints(lm){
  const eyeMid = mid(lm[IDX.leftEyeOuter], lm[IDX.rightEyeOuter]);
  const eyeDist = Math.max(dist(lm[IDX.leftEyeOuter], lm[IDX.rightEyeOuter]), 1e-5);
  const left = lm[IDX.leftEyeOuter], right = lm[IDX.rightEyeOuter];
  const angle = Math.atan2(right.y-left.y, right.x-left.x);
  const cos=Math.cos(-angle), sin=Math.sin(-angle);
  return lm.map(p=>{
    const x=p.x-eyeMid.x, y=p.y-eyeMid.y;
    return {x:(x*cos-y*sin)/eyeDist,y:(x*sin+y*cos)/eyeDist,z:(p.z||0)/eyeDist};
  });
}
function featureVector(lm){
  const n=normalizedPoints(lm);
  const pick=(arr)=>arr.flatMap(i=>[n[i].x,n[i].y]);
  const le=mid(n[IDX.leftEyeOuter],n[IDX.leftEyeInner]);
  const re=mid(n[IDX.rightEyeOuter],n[IDX.rightEyeInner]);
  const eyeDistance=dist(le,re);
  const faceWidth=dist(n[IDX.leftCheek],n[IDX.rightCheek]);
  const faceHeight=dist(n[IDX.forehead],n[IDX.chin]);
  const noseWidth=dist(n[IDX.noseLeft],n[IDX.noseRight]);
  const noseLen=dist(n[IDX.noseTip],n[IDX.forehead]);
  const mouthWidth=dist(n[IDX.mouthL],n[IDX.mouthR]);
  const noseMouth=dist(n[IDX.noseTip],mid(n[IDX.mouthTop],n[IDX.mouthBottom]));
  const browNose=dist(mid(n[IDX.leftBrow],n[IDX.rightBrow]),n[IDX.noseTip]);
  const noseChin=dist(n[IDX.noseTip],n[IDX.chin]);
  const eyeHeights=[dist(n[IDX.leftUpper],n[IDX.leftLower]),dist(n[IDX.rightUpper],n[IDX.rightLower])];
  const asymmetry=Math.abs(eyeHeights[0]-eyeHeights[1])+Math.abs(n[IDX.leftCheek].y-n[IDX.rightCheek].y);
  return {
    contour:pick(contourIdx), eyes:pick(eyeLIdx.concat(eyeRIdx)), nose:pick(noseIdx), mouth:pick(mouthIdx), brow:pick(browIdx),
    ratios:[faceHeight/faceWidth,eyeDistance/faceWidth,noseWidth/faceWidth,noseLen/faceHeight,mouthWidth/faceWidth,noseMouth/faceHeight,browNose/faceHeight,noseChin/faceHeight],
    asymmetry
  };
}
function vecDistance(a,b){let sum=0,n=0; for(let i=0;i<a.length;i++){const d=a[i]-b[i];sum+=d*d;n++;} return Math.sqrt(sum/Math.max(n,1));}
function similarity(a,b){return clamp(100*Math.exp(-3.0*vecDistance(a,b)));}
function ratioSimilarity(a,b){let dif=0;for(let i=0;i<a.length;i++){dif+=Math.min(Math.abs(a[i]-b[i])*2.8,1);}return clamp(100*(1-dif/a.length));}

function computeFaceScores(a,b){
  const A=featureVector(a),B=featureVector(b);
  const shape=similarity(A.contour,B.contour)*.72 + ratioSimilarity(A.ratios.slice(0,1),B.ratios.slice(0,1))*.28;
  const eyes=similarity(A.eyes,B.eyes)*.82 + ratioSimilarity(A.ratios.slice(1,2),B.ratios.slice(1,2))*.18;
  const nose=similarity(A.nose,B.nose)*.8 + ratioSimilarity(A.ratios.slice(2,4),B.ratios.slice(2,4))*.2;
  const mouth=similarity(A.mouth,B.mouth)*.75 + ratioSimilarity(A.ratios.slice(4),B.ratios.slice(4))*.25;
  const asym=clamp(100-Math.abs(A.asymmetry-B.asymmetry)*110);
  const face=shape*.22+eyes*.28+nose*.18+mouth*.18+asym*.14;
  return {shape,eyes,nose,mouth,face:clamp(face)};
}

function drawLandmarks(canvas,img,lm,visible=true){
  const ctx=canvas.getContext('2d');
  canvas.width=img.naturalWidth||img.width; canvas.height=img.naturalHeight||img.height;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if(!visible||!lm)return;
  ctx.lineWidth=Math.max(1,canvas.width/700); ctx.strokeStyle='rgba(255,218,104,.45)'; ctx.fillStyle='rgba(255,75,159,.8)';
  const groups=[contourIdx,eyeLIdx,eyeRIdx,noseIdx,mouthIdx,browIdx];
  groups.forEach((group,gi)=>{
    ctx.beginPath(); group.forEach((i,k)=>{const p=lm[i];const x=p.x*canvas.width,y=p.y*canvas.height;k?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();
    group.forEach(i=>{const p=lm[i];ctx.beginPath();ctx.arc(p.x*canvas.width,p.y*canvas.height,Math.max(1.4,canvas.width/430),0,Math.PI*2);ctx.fill();});
  });
}
function resizeCanvasToImage(canvas,img){canvas.style.aspectRatio=`${img.naturalWidth}/${img.naturalHeight}`;canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;}

function estimateExposure(img){
  // Game-only heuristic: estimate visible skin-like pixels in the non-face image area.
  // It never uploads the image. This intentionally avoids claiming medical/identity accuracy.
  const maxSide=420, scale=Math.min(1,maxSide/Math.max(img.naturalWidth,img.naturalHeight));
  const w=Math.max(1,Math.round(img.naturalWidth*scale)),h=Math.max(1,Math.round(img.naturalHeight*scale));
  const c=document.createElement('canvas');c.width=w;c.height=h;const ctx=c.getContext('2d',{willReadFrequently:true});ctx.drawImage(img,0,0,w,h);
  const data=ctx.getImageData(0,0,w,h).data; let skin=0,personish=0;
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){
    const i=(y*w+x)*4,r=data[i],g=data[i+1],b=data[i+2];
    const mx=Math.max(r,g,b),mn=Math.min(r,g,b),sat=mx-mn;
    const looksSkin=r>85 && g>45 && b>35 && r>g*1.08 && g>b*1.02 && sat>18 && r-g<105;
    const nonDark=mx>45;
    if(looksSkin)skin++; if(nonDark)personish++;
  }
  const raw=skin/Math.max(personish,1);
  return clamp(Math.round(Math.pow(clamp((raw-0.04)/0.30,0,1),0.78)*30),0,30);
}

function setAnalysis(title,text,pct){els.analysisTitle.textContent=title;els.analysisText.textContent=text;els.progressBar.style.width=`${pct}%`;}
function toast(msg){els.toast.textContent=msg;els.toast.classList.add('show');clearTimeout(toast.t);toast.t=setTimeout(()=>els.toast.classList.remove('show'),2200);}
function sparks(score){const count=score>=95?42:score>=85?24:score>=70?12:4;for(let i=0;i<count;i++){const s=document.createElement('div');s.className='spark '+(i%3===0?'star':'');s.style.left=(30+Math.random()*40)+'%';s.style.top=(20+Math.random()*45)+'%';s.style.setProperty('--dx',(Math.random()*260-130)+'px');s.style.setProperty('--dy',(Math.random()*-250-40)+'px');els.sparkLayer.appendChild(s);setTimeout(()=>s.remove(),1400)}}
function verdict(score){if(score>=97)return ['GOLDEN MATCH!!!','AIも困惑。ほぼ本人。'];if(score>=90)return ['本人級！！！','測定器がかなり喜んでいます。'];if(score>=80)return ['かなり似てる！！','顔面一致度、高め。'];if(score>=65)return ['そこそこ一致','似ている要素はあります。'];if(score>=45)return ['微妙に一致','測定器は悩んでいます。'];return ['別人判定寄り','顔面測定器、静かに首をかしげる。'];}

// 現行のスコアをかなり厳しく再マッピングする。
// 70点→40点、90点→97点、100点→100点となるように、
// 中〜高得点域を極端に引き締める。
function strictScore(raw){
  raw=clamp(raw);
  if(raw<=70) return Math.round(raw*(40/70));
  if(raw<=90) return Math.round(40+(raw-70)*(57/20));
  return Math.round(97+(raw-90)*0.3);
}

async function initFaceMesh(){
  if(typeof FaceMesh==='undefined')throw new Error('MediaPipe Face Mesh が読み込めませんでした。');
  faceMesh=new FaceMesh({locateFile:(file)=>`https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/${file}`});
  faceMesh.setOptions({maxNumFaces:2,refineLandmarks:true,minDetectionConfidence:.55,minTrackingConfidence:.55});
  await new Promise((resolve,reject)=>{
    let settled=false;
    faceMesh.onResults((res)=>{faceMesh._lastResults=res;if(!settled){settled=true;resolve(res)}});
    faceMesh._readyReject=reject;
    setTimeout(()=>{if(!settled){settled=true;reject(new Error('AIモデルの初期化がタイムアウトしました。'))}},20000);
    const blank=document.createElement('canvas');blank.width=2;blank.height=2;faceMesh.send({image:blank}).catch(reject);
  });
}
async function detect(img){
  faceMesh._lastResults=null;
  await faceMesh.send({image:img});
  const res=faceMesh._lastResults;
  if(!res||!res.multiFaceLandmarks)return [];
  return res.multiFaceLandmarks;
}

async function loadTarget(){
  try{
    setAnalysis('AIモデル起動中……','顔面測定器の電源を入れています。',15);
    await initFaceMesh();
    setAnalysis('TARGET FACEをスキャン中……','基準顔のランドマークを取得しています。',35);
    const faces=await detect(els.targetImage);
    if(faces.length!==1)throw new Error(faces.length===0?'基準顔が見つかりませんでした。':'基準画像に複数の顔があります。');
    targetLandmarks=faces[0];targetImageReady=true;resizeCanvasToImage(els.targetCanvas,els.targetImage);drawLandmarks(els.targetCanvas,els.targetImage,targetLandmarks,showTargetLandmarks);
    setAnalysis('TARGET LOCKED.','基準顔を100点の比較基準としてロックしました。',100);
    els.fileStatus.textContent='READY';els.fileStatus.classList.add('ready');
  }catch(err){setAnalysis('AI起動失敗！','ブラウザでAIモデルを読み込めませんでした。ネット接続とローカルサーバーを確認してください。',0);toast(err.message);}
}

async function processTest(file){
  if(!targetImageReady){toast('基準顔のAI解析がまだ終わっていません。');return;}
  const ok=/^image\/(jpeg|png|webp)$/.test(file.type)||/\.(jpe?g|png|webp)$/i.test(file.name);
  if(!ok){toast('その画像形式は測定器が対応していません！');return;}
  if(file.size>15*1024*1024){toast('画像が大きすぎます。15MB以下にしてください。');return;}
  if(testObjectUrl)URL.revokeObjectURL(testObjectUrl);testObjectUrl=URL.createObjectURL(file);
  els.testImage.src=testObjectUrl;els.testPreview.classList.remove('hidden');els.dropZone.classList.add('hidden');els.fileStatus.textContent='SCANNING';
  await new Promise((resolve,reject)=>{els.testImage.onload=resolve;els.testImage.onerror=reject});
  try{
    setAnalysis('顔面解析中……','目の位置を確認しています……',25);await wait(350);
    const faces=await detect(els.testImage);
    if(faces.length===0)throw new Error('顔が見つかりませんでした！顔面をもう少し大きくしてください。');
    if(faces.length>1)throw new Error('顔面を1人にしてください！測定器は2人同時に処理できません。');
    testLandmarks=faces[0];resizeCanvasToImage(els.testCanvas,els.testImage);drawLandmarks(els.testCanvas,els.testImage,testLandmarks,showTestLandmarks);
    setAnalysis('特徴量を比較中……','基準顔との距離を計算しています……',55);await wait(350);
    const scores=computeFaceScores(targetLandmarks,testLandmarks);
    setAnalysis('身体領域ボーナス計算中……','画像内の視認可能領域をスキャンしています……',75);await wait(350);
    const exposure=estimateExposure(els.testImage);
    const face70=Math.round(scores.face*.7);
    const rawTotal=clamp(face70+exposure,0,100);
    const total=strictScore(rawTotal);
    showResult(scores,face70,exposure,total,rawTotal);
  }catch(err){setAnalysis('AIチェック',err.message||'AI解析を完了できませんでした。',0);toast(err.message||'解析を完了できませんでした。');}
}
function showResult(scores,face70,exposure,total,rawTotal){
  els.faceScore.textContent=face70;els.exposureScore.textContent=exposure;els.totalScore.textContent=total;
  els.shapeScore.textContent=Math.round(scores.shape)+'%';els.eyeScore.textContent=Math.round(scores.eyes)+'%';els.noseScore.textContent=Math.round(scores.nose)+'%';els.mouthScore.textContent=Math.round(scores.mouth)+'%';
  const [v,sub]=verdict(total);els.verdict.textContent=v+' — '+sub;
  els.resultSection.classList.remove('hidden');els.controls.classList.remove('hidden');
  document.body.classList.toggle('high',total>=80);
  els.meterBar.style.height=`${Math.max(3,total)}%`;
  els.meterRank.textContent=total>=95?'ULTRA RARE':total>=85?'GOLD':'NORMAL';
  els.meterCaption.textContent=total>=95?'測定器、金色の何かを感じています。':total>=80?'ぐいーん！！かなり伸びました。':total>=60?'測定器、ちょっと嬉しそう。':'測定器は静かです。';
  els.meterFlare.classList.toggle('active',total>=80);
  els.meterDevice.classList.toggle('balls-lit',total>=70);
  setAnalysis('測定完了！！',`${v}　顔・露出の評価を算出しました。`,100);
  setTimeout(()=>els.resultSection.scrollIntoView({behavior:'smooth',block:'start'}),150);
  if(total>=70)sparks(total);
}

function reset(){
  if(testObjectUrl)URL.revokeObjectURL(testObjectUrl);testObjectUrl=null;testLandmarks=null;els.testImage.removeAttribute('src');els.testPreview.classList.add('hidden');els.dropZone.classList.remove('hidden');els.fileInput.value='';els.resultSection.classList.add('hidden');els.controls.classList.add('hidden');els.fileStatus.textContent='WAITING';els.faceScore.textContent='--';els.totalScore.textContent='--';els.exposureScore.textContent='0';els.meterBar.style.height='0';els.meterFlare.classList.remove('active');els.meterDevice.classList.remove('balls-lit');document.body.classList.remove('high');setAnalysis('測定器、待機中。','比較画像を投入すると顔面スキャンを開始します。',100);window.scrollTo({top:0,behavior:'smooth'});}

els.chooseBtn.addEventListener('click',(e)=>{e.preventDefault();els.fileInput.click()});els.fileInput.addEventListener('change',e=>{if(e.target.files[0])processTest(e.target.files[0])});
['dragenter','dragover'].forEach(ev=>els.dropZone.addEventListener(ev,e=>{e.preventDefault();els.dropZone.classList.add('drag')}));['dragleave','drop'].forEach(ev=>els.dropZone.addEventListener(ev,e=>{e.preventDefault();els.dropZone.classList.remove('drag')}));els.dropZone.addEventListener('drop',e=>{const f=e.dataTransfer.files[0];if(f)processTest(f)});
els.resetBtn.addEventListener('click',reset);
els.landmarkToggle.addEventListener('click',()=>{showTestLandmarks=!showTestLandmarks;els.landmarkToggle.textContent=showTestLandmarks?'ON':'OFF';els.landmarkToggle.classList.toggle('on',showTestLandmarks);if(testLandmarks)drawLandmarks(els.testCanvas,els.testImage,testLandmarks,showTestLandmarks)});

window.addEventListener('error',e=>{if(/FaceMesh|face_mesh|mediapipe/i.test(e.message||''))toast('AIライブラリの読み込みに失敗しました。');});
if(els.targetImage.complete)loadTarget();else els.targetImage.addEventListener('load',loadTarget,{once:true});
