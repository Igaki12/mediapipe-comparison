// Copyright 2023 The MediaPipe Authors.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
import {
    PoseLandmarker,
    FilesetResolver,
    DrawingUtils
} from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";

const demosSection = document.getElementById("demos");

let poseLandmarker = undefined;
let runningMode = "IMAGE";
let imageSegmenter;
// Before we can use PoseLandmarker class we must wait for it to finish
// loading. Machine Learning models can be large and take a moment to
// get everything needed to run.
const loadingInterval = setInterval(() => {
    document.getElementById("loadingMsg").innerHTML += ".";
    if ( document.getElementById("loadingMsg").innerHTML.length > 25) {
        document.getElementById("loadingMsg").innerHTML = "モジュールの読み込みに数秒かかります...";
    }
    },1000);

const createPoseLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
            delegate: "GPU"
        },
        runningMode: runningMode,
        numPoses: 2
    });
    demosSection.classList.remove("invisible");
    document.getElementById("loadingMsg").style.display = "none";
    clearInterval(loadingInterval);
    runPoseEstimation();
};
createPoseLandmarker();

const positionNamesJP = [
    "鼻 (nose)",
    "左目-内側 (left eye - inner)",
    "左目 (left eye)",
    "左目-外側 (left eye - outer)",
    "右目-内側 (right eye - inner)",
    "右目 (right eye)",
    "右目-外側 (right eye - outer)",
    "左耳 (left ear)",
    "右耳 (right ear)",
    "口-左縁 (mouth - left)",
    "口-右縁 (mouth - right)",
    "左肩 (left shoulder)",
    "右肩 (right shoulder)",
    "左肘 (left elbow)",
    "右肘 (right elbow)",
    "左手首 (left wrist)",
    "右手首 (right wrist)",
    "左小指 (left pinky)",
    "右小指 (right pinky)",
    "左人差し指 (left index)",
    "右人差し指 (right index)",
    "左親指 (left thumb)",
    "右親指 (right thumb)",
    "左腰 (left hip)",
    "右腰 (right hip)",
    "左膝 (left knee)",
    "右膝 (right knee)",
    "左足首 (left ankle)",
    "右足首 (right ankle)",
    "左かかと (left heel)",
    "右かかと (right heel)",
    "左足先 (left foot index)",
    "右足先 (right foot index)"
];





/********************************************************************
// Demo 1: Grab a bunch of images from the page and detection them
// upon click.
********************************************************************/

// In this demo, we have put all our clickable images in divs with the
// CSS class 'detectionOnClick'. Lets get all the elements that have
// this class.

// すでにアップされている画像を選択して表示し、ポーズ推定を行う
let result_before = [];
const image_before = document.getElementById("image_before");
const canvas_overlay = document.getElementById("canvas_overlay");
const canvas_before = document.getElementById("canvas_before");
canvas_overlay.width = image_before.width;
canvas_overlay.height = image_before.height;
canvas_overlay.style.top = image_before.offsetTop;
canvas_overlay.style.left = 0;
canvas_before.width = image_before.width;
canvas_before.height = image_before.height;
// 画像が表示されたら、ポーズ推定を行う
const runPoseEstimation = () => {
// image_before.onload = () => {
    console.log("image_before.onload");
    if (!poseLandmarker) {
        console.log("Wait for poseLandmarker to load before clicking!");
        return;
    }

    if (runningMode === "VIDEO") {
        runningMode = "IMAGE";
        poseLandmarker.setOptions({ runningMode: "IMAGE" });
    }
    // poseLandmarker.setOptions({ outputSegmentationMasks: true });
    poseLandmarker.detect(image_before, async (result) => {
        const canvas_overlayCtx = canvas_overlay.getContext("2d");
        const canvas_beforeCtx = canvas_before.getContext("2d");
        const drawingUtils_overlay = new DrawingUtils(canvas_overlayCtx);
        const drawingUtils_before = new DrawingUtils(canvas_beforeCtx);
        for (const landmark of result.landmarks) {
            drawingUtils_overlay.drawLandmarks(landmark, {
                radius: (data) => DrawingUtils.lerp(data.from?.z ?? 0, -0.15, 0.1, 5, 1),
                lineWidth: 2,
            });
            drawingUtils_before.drawLandmarks(landmark, {
                radius: (data) => DrawingUtils.lerp(data.from?.z ?? 0, -0.15, 0.1, 5, 1),
                lineWidth: 2,
            });
            drawingUtils_overlay.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS,
                { lineWidth: 2, color: "white" });
            drawingUtils_before.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS,
                { lineWidth: 2, color: "white" });
        }

        
        console.log("result : ");
        console.log(result);
        result_before = result;
        document.getElementById("fileSelector").disabled = false;
        document.getElementById("loadingMsg2").innerText = "比較したい画像を選択してください▽";

    })};


// 手持ちの画像を選択して表示し、ポーズ推定を行う

const FileSelector = document.getElementById("fileSelector");
const image_after = document.getElementById("selectedImage");
const canvas_after = document.getElementById("canvas_after");
canvas_after.width = image_after.width;
canvas_after.height = image_after.height;
canvas_after.style.top = image_after.offsetTop;
canvas_after.style.left = image_after.offsetLeft;
let result_after = [];
FileSelector.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (!file) {
        return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
        image_after.src = e.target.result;
    };
    reader.readAsDataURL(file);
    // 画像が表示されたら、ポーズ推定を行う
    image_after.onload = () => {
        if (!poseLandmarker) {
            console.log("Wait for poseLandmarker to load before clicking!");
            return;
        }

        if (runningMode === "VIDEO") {
            runningMode = "IMAGE";
            poseLandmarker.setOptions({ runningMode: "IMAGE" });
        }
        poseLandmarker.detect(image_after, async (result) => {
            const canvas_afterCtx = canvas_after.getContext("2d");
            const drawingUtils_after = new DrawingUtils(canvas_afterCtx);
            for (const landmark of result.landmarks) {
                drawingUtils_after.drawLandmarks(landmark, {
                    radius: (data) => DrawingUtils.lerp(data.from?.z ?? 0, -0.15, 0.1, 5, 1)
                });
                drawingUtils_after.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
            }
            if(result_before.length > 0 && result_before.landmarks.length > 0){
                for (const landmark of result_before.landmarks) {
                    drawingUtils_after.drawLandmarks(landmark, {
                        radius: (data) => DrawingUtils.lerp(data.from?.z ?? 0, -0.15, 0.1, 5, 1),
                        color: "orange",
                        lineWidth: 2,
                    });
                    drawingUtils_after.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS,
                        { lineWidth: 2, color: "orange" });
                }
            }
            console.log("canvas_after_result : ");
            console.log(result);
            result_after = result;
        });

    };
});