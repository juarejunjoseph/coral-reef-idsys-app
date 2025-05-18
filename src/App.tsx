import React, { useEffect, useRef, useState } from 'react';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import '@tensorflow/tfjs';
import { Camera, RefreshCcw, Camera as Camera2 } from 'lucide-react';

interface Detection {
  label: string;
  confidence: number;
  type: 'object' | 'scene';
}

function App() {
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [mobilenetModel, setMobilenetModel] = useState<mobilenet.MobileNet | null>(null);
  const [cocoModel, setCocoModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [isFrontCamera, setIsFrontCamera] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    const loadModels = async () => {
      const [loadedMobilenet, loadedCoco] = await Promise.all([
        mobilenet.load(),
        cocoSsd.load()
      ]);
      setMobilenetModel(loadedMobilenet);
      setCocoModel(loadedCoco);
    };
    loadModels();
  }, []);

  useEffect(() => {
    const requestCameraPermission = async () => {
      try {
        const constraints = {
          video: {
            facingMode: isFrontCamera ? 'user' : 'environment',
            width: { ideal: 1080 },
            height: { ideal: 1920 }
          }
        };
        
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        setHasPermission(true);
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        setHasPermission(false);
        console.error('Failed to get camera permission:', err);
      }
    };

    requestCameraPermission();

    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, [isFrontCamera]);

  const processFrame = async () => {
    if (!mobilenetModel || !cocoModel || !videoRef.current) return;

    const [sceneResults, objectResults] = await Promise.all([
      mobilenetModel.classify(videoRef.current),
      cocoModel.detect(videoRef.current)
    ]);

    const combinedDetections: Detection[] = [
      ...sceneResults.map(result => ({
        label: result.className,
        confidence: result.probability,
        type: 'scene' as const
      })),
      ...objectResults.map(result => ({
        label: result.class,
        confidence: result.score,
        type: 'object' as const
      }))
    ];

    const topDetections = combinedDetections
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 5);

    setDetections(topDetections);
  };

  useEffect(() => {
    const interval = setInterval(processFrame, 1000);
    return () => clearInterval(interval);
  }, [mobilenetModel, cocoModel]);

  const toggleCamera = async () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    setIsFrontCamera(!isFrontCamera);
  };

  if (hasPermission === null) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-xl">Initializing ANN...</div>
      </div>
    );
  }

  if (hasPermission === false) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="bg-white p-8 rounded-lg shadow-md max-w-md w-full mx-4">
          <div className="flex items-center justify-center mb-6">
            <Camera className="w-16 h-16 text-red-500" />
          </div>
          <h2 className="text-2xl font-bold text-center mb-4">Camera Access Required</h2>
          <p className="text-gray-600 text-center">
            Please enable camera access in your browser settings to use this application.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100">
      <div className="fixed inset-0 flex flex-col">
        <div className="bg-white shadow-sm p-4">
          <h2 className="text-xl font-semibold">Coral Reef ID Software System</h2>
        </div>
        
        <div className="flex-1 relative">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            className="absolute inset-0 w-full h-full object-cover"
            style={{ transform: isFrontCamera ? 'scaleX(-1)' : 'none' }}
          />
          
          <div className="absolute top-4 right-4 flex gap-2">
            <button
              onClick={toggleCamera}
              className="p-2 bg-white/80 rounded-full hover:bg-white transition-colors"
              title="Switch camera"
            >
              <Camera2 className="w-6 h-6" />
            </button>
            <button
              onClick={() => setDetections([])}
              className="p-2 bg-white/80 rounded-full hover:bg-white transition-colors"
              title="Reset detections"
            >
              <RefreshCcw className="w-6 h-6" />
            </button>
          </div>
        </div>
        
        <div className="bg-white shadow-lg rounded-t-xl p-4 max-h-[40vh] overflow-y-auto">
          <h2 className="text-xl font-semibold mb-3">Detected Items:</h2>
          <div className="space-y-2">
            {detections.map((detection, index) => (
              <div
                key={index}
                className={`p-3 rounded-md ${
                  detection.type === 'object' 
                    ? 'bg-blue-50 text-blue-700' 
                    : 'bg-green-50 text-green-700'
                }`}
              >
                <div className="flex justify-between items-center">
                  <span className="font-medium">{detection.label}</span>
                  <span className="text-sm">
                    {(detection.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="text-sm opacity-75 capitalize">
                  Type: {detection.type}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;