"use client";

import React, { useState, useRef, useCallback, useEffect } from "react";
import {
  Upload,
  Type,
  Square,
  SquareMinus,
  Trash2,
  Loader2,
  CheckCircle2,
  XCircle,
  Sparkles,
  BoxSelect,
  Timer,
  Server,
  MousePointer2,
  MousePointerClick,
  Stethoscope,
  Lightbulb,
  Layers,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { SegmentationCanvas } from "@/components/SegmentationCanvas";
import {
  uploadImage,
  segmentWithText,
  addBoxPrompt,
  addPointPrompt,
  resetPrompts,
  checkHealth,
  getModalities,
  setModality,
  changeSlice,
  propagateMasks,
  getVolumeMask,
  type SegmentationResult,
  type ModalityConfig,
} from "@/lib/api";

type BoxMode = "positive" | "negative";
type PointMode = "positive" | "negative";
type InteractionMode = "box" | "point";

interface TimingEntry {
  label: string;
  duration: number;
  timestamp: Date;
}

function formatDuration(ms: number | undefined | null): string {
  if (ms === undefined || ms === null || isNaN(ms)) {
    return "—";
  }
  if (ms < 1000) {
    return `${Math.round(ms)}ms`;
  }
  return `${(ms / 1000).toFixed(2)}s`;
}

export default function Home() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageWidth, setImageWidth] = useState(0);
  const [imageHeight, setImageHeight] = useState(0);
  const [result, setResult] = useState<SegmentationResult | null>(null);
  const [textPrompt, setTextPrompt] = useState("");
  const [boxMode, setBoxMode] = useState<BoxMode>("positive");
  const [pointMode, setPointMode] = useState<PointMode>("positive");
  const [interactionMode, setInteractionMode] = useState<InteractionMode>("box");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<
    "checking" | "online" | "offline"
  >("checking");
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Volume/DICOM state
  const [totalSlices, setTotalSlices] = useState(1);
  const [currentSlice, setCurrentSlice] = useState(0);
  const [isVolume, setIsVolume] = useState(false);
  const [isDicom, setIsDicom] = useState(false);
  const [propagatedSlices, setPropagatedSlices] = useState<number[]>([]);

  // Medical imaging state
  const [modalities, setModalities] = useState<string[]>([]);
  const [selectedModality, setSelectedModality] = useState<string>("general");
  const [modalityConfig, setModalityConfig] = useState<ModalityConfig | null>(null);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);

  // Timing state (server-side processing times)
  const [timings, setTimings] = useState<TimingEntry[]>([]);
  const [lastTiming, setLastTiming] = useState<TimingEntry | null>(null);

  const addTiming = useCallback(
    (label: string, duration: number | undefined | null) => {
      // Only add timing if we have a valid duration from the server
      if (duration === undefined || duration === null || isNaN(duration)) {
        console.warn(`No timing data for: ${label}`);
        return;
      }
      const entry: TimingEntry = { label, duration, timestamp: new Date() };
      setLastTiming(entry);
      setTimings((prev) => [entry, ...prev].slice(0, 10)); // Keep last 10
    },
    []
  );

  // Check backend health on mount and load modalities
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const health = await checkHealth();
        setBackendStatus(health.model_loaded ? "online" : "checking");
        
        // Load modalities
        if (health.model_loaded) {
          try {
            const modalitiesData = await getModalities();
            setModalities(modalitiesData.modalities);
            if (modalitiesData.configs.general) {
              setModalityConfig(modalitiesData.configs.general);
            }
          } catch (err) {
            console.error("Failed to load modalities:", err);
          }
        }
      } catch {
        setBackendStatus("offline");
      }
    };
    checkBackend();
    const interval = setInterval(checkBackend, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleFileSelect = useCallback(
    async (files: FileList) => {
      setError(null);
      setIsLoading(true);

      try {
        // Upload to backend (supports multiple files)
        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
          formData.append('files', files[i]);
        }
        
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/upload`,
          {
            method: "POST",
            body: formData,
          }
        );
        
        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.detail || "Upload failed");
        }
        
        const data = await response.json();
        
        setSessionId(data.session_id);
        setImageWidth(data.width);
        setImageHeight(data.height);
        setTotalSlices(data.total_slices);
        setCurrentSlice(data.current_slice);
        setIsVolume(data.is_volume);
        setIsDicom(data.is_dicom);
        setResult(null);
        setTextPrompt("");
        setPropagatedSlices([]);
        
        // Set image URL based on type
        if (data.is_dicom) {
          // For DICOM, use backend endpoint to get rendered image
          const imageUrl = `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/slice-image/${data.session_id}?slice_index=${data.current_slice}`;
          setImageUrl(imageUrl);
        } else {
          // For regular images, create object URL
          const url = URL.createObjectURL(files[0]);
          setImageUrl(url);
        }
        
        addTiming("Image Encoding", data.processing_time_ms);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to upload file");
        setImageUrl(null);
      } finally {
        setIsLoading(false);
      }
    },
    [addTiming]
  );

  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      
      const items = e.dataTransfer.items;
      const files: File[] = [];
      
      // Process all dropped items
      if (items) {
        for (let i = 0; i < items.length; i++) {
          const item = items[i];
          
          if (item.kind === 'file') {
            const entry = item.webkitGetAsEntry?.();
            
            if (entry?.isDirectory) {
              // It's a directory - read all files recursively
              await readDirectory(entry as FileSystemDirectoryEntry, files);
            } else {
              // It's a file
              const file = item.getAsFile();
              if (file) files.push(file);
            }
          }
        }
      } else {
        // Fallback to simple file list
        const fileList = e.dataTransfer.files;
        for (let i = 0; i < fileList.length; i++) {
          files.push(fileList[i]);
        }
      }
      
      if (files.length > 0) {
        console.log(`Dropped ${files.length} files`);
        const fileList = createFileList(files);
        handleFileSelect(fileList);
      }
    },
    [handleFileSelect]
  );
  
  // Helper to read directory recursively
  const readDirectory = async (entry: FileSystemDirectoryEntry, files: File[]) => {
    const reader = entry.createReader();
    
    return new Promise<void>((resolve) => {
      const readEntries = () => {
        reader.readEntries(async (entries) => {
          if (entries.length === 0) {
            resolve();
            return;
          }
          
          for (const entry of entries) {
            if (entry.isFile) {
              const file = await new Promise<File>((resolve) => {
                (entry as FileSystemFileEntry).file(resolve);
              });
              files.push(file);
            } else if (entry.isDirectory) {
              await readDirectory(entry as FileSystemDirectoryEntry, files);
            }
          }
          
          readEntries(); // Continue reading
        });
      };
      
      readEntries();
    });
  };
  
  // Helper to create FileList from File array
  const createFileList = (files: File[]): FileList => {
    const dataTransfer = new DataTransfer();
    files.forEach(file => dataTransfer.items.add(file));
    return dataTransfer.files;
  };

  const handleTextSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!sessionId || !textPrompt.trim()) return;

    setError(null);
    setIsLoading(true);
    setShowSuggestions(false);

    try {
      const response = await segmentWithText(sessionId, textPrompt.trim());
      setResult(response.results);
      addTiming(`Text: "${textPrompt.trim()}"`, response.processing_time_ms);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Segmentation failed");
    } finally {
      setIsLoading(false);
    }
  };

  const handleModalityChange = async (modality: string) => {
    if (!sessionId) {
      setSelectedModality(modality);
      return;
    }

    setError(null);
    setIsLoading(true);

    try {
      const response = await setModality(sessionId, modality);
      setSelectedModality(modality);
      setModalityConfig(response.config);
      setSuggestions(response.suggestions);
      setShowSuggestions(true);
      addTiming(`Set Modality: ${modality}`, 0);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to set modality");
    } finally {
      setIsLoading(false);
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setTextPrompt(suggestion);
    setShowSuggestions(false);
  };

  const handleBoxDrawn = useCallback(
    async (box: number[]) => {
      if (!sessionId) return;

      setError(null);
      setIsLoading(true);

      try {
        const response = await addBoxPrompt(
          sessionId,
          box,
          boxMode === "positive"
        );
        setResult(response.results);
        addTiming(`Box (${boxMode})`, response.processing_time_ms);
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Failed to add box prompt"
        );
      } finally {
        setIsLoading(false);
      }
    },
    [sessionId, boxMode, addTiming]
  );

  const handlePointClicked = useCallback(
    async (point: number[]) => {
      if (!sessionId) return;

      setError(null);
      setIsLoading(true);

      try {
        const response = await addPointPrompt(
          sessionId,
          point,
          pointMode === "positive"
        );
        setResult(response.results);
        addTiming(`Point (${pointMode})`, response.processing_time_ms);
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Failed to add point prompt"
        );
      } finally {
        setIsLoading(false);
      }
    },
    [sessionId, pointMode, addTiming]
  );

  const handleReset = async () => {
    if (!sessionId) return;

    setError(null);
    setIsLoading(true);

    try {
      const response = await resetPrompts(sessionId);
      setResult(response.results);
      setTextPrompt("");
      addTiming("Reset Prompts", response.processing_time_ms);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to reset");
    } finally {
      setIsLoading(false);
    }
  };

  const handleSliceChange = useCallback(
    async (newSliceIndex: number) => {
      if (!sessionId || newSliceIndex === currentSlice) return;

      setError(null);
      setIsLoading(true);

      try {
        const response = await changeSlice(sessionId, newSliceIndex);
        setCurrentSlice(newSliceIndex);
        
        // Check if we have a propagated mask for this slice
        if (propagatedSlices.includes(newSliceIndex)) {
          try {
            const maskResponse = await getVolumeMask(sessionId, newSliceIndex);
            setResult(maskResponse.results);
          } catch {
            setResult(response.results);
          }
        } else {
          setResult(response.results);
        }
        
        setImageWidth(response.width || imageWidth);
        setImageHeight(response.height || imageHeight);
        
        // Update image URL for DICOM
        if (isDicom) {
          const imageUrl = `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/slice-image/${sessionId}?slice_index=${newSliceIndex}&t=${Date.now()}`;
          setImageUrl(imageUrl);
        }
        
        addTiming(`Slice ${newSliceIndex + 1}`, response.processing_time_ms);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to change slice");
      } finally {
        setIsLoading(false);
      }
    },
    [sessionId, currentSlice, imageWidth, imageHeight, isDicom, addTiming, propagatedSlices]
  );

  const handlePropagate = useCallback(
    async () => {
      if (!sessionId || !isVolume) return;

      setError(null);
      setIsLoading(true);

      try {
        const response = await propagateMasks(sessionId, "both");
        setPropagatedSlices(response.propagated_slices);
        addTiming(`Propagate (${response.total_propagated} slices)`, response.processing_time_ms);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to propagate masks");
      } finally {
        setIsLoading(false);
      }
    },
    [sessionId, isVolume, addTiming]
  );

  const maskCount = result?.masks?.length ?? 0;

  // Calculate average inference time (excluding upload)
  const inferenceTimings = timings.filter(
    (t) => !t.label.includes("Upload") && !t.label.includes("Reset")
  );
  const avgInferenceTime =
    inferenceTimings.length > 0
      ? inferenceTimings.reduce((sum, t) => sum + t.duration, 0) /
        inferenceTimings.length
      : null;

  return (
    <main className="min-h-screen p-6 md:p-8">
      {/* Header */}
      <header className="max-w-7xl mx-auto mb-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-primary/20 rounded-lg pulse-glow">
              <Stethoscope className="w-6 h-6 text-primary" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight">MedSAM3 Studio</h1>
              <p className="text-sm text-muted-foreground">
                Medical image segmentation with LoRA fine-tuning
              </p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            {/* Backend status */}
            {backendStatus === "checking" && (
              <div className="flex items-center gap-2 text-muted-foreground text-sm">
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Connecting...</span>
              </div>
            )}
            {backendStatus === "online" && (
              <div className="flex items-center gap-2 text-primary text-sm">
                <CheckCircle2 className="w-4 h-4" />
                <span>Model Ready</span>
              </div>
            )}
            {backendStatus === "offline" && (
              <div className="flex items-center gap-2 text-destructive text-sm">
                <XCircle className="w-4 h-4" />
                <span>Backend Offline</span>
              </div>
            )}
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-[340px_1fr] gap-6">
        {/* Sidebar Controls */}
        <aside className="space-y-4">
          {/* Upload Card */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center gap-2">
                <Upload className="w-4 h-4" />
                Image Source
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div
                onClick={() => fileInputRef.current?.click()}
                onDrop={handleDrop}
                onDragOver={(e) => e.preventDefault()}
                className="border-2 border-dashed border-border rounded-lg p-6 text-center hover:border-primary/50 hover:bg-primary/5 transition-all cursor-pointer"
              >
                <Upload className="w-8 h-8 mx-auto mb-2 text-muted-foreground" />
                <p className="text-sm text-muted-foreground">
                  Click or drop image/DICOM here
                </p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*,.dcm,.dicom"
                  multiple
                  onChange={(e) => {
                    const files = e.target.files;
                    if (files && files.length > 0) handleFileSelect(files);
                  }}
                  className="hidden"
                />
              </div>
              {imageWidth > 0 && (
                <div className="mt-2 space-y-1">
                  <p className="text-xs text-muted-foreground text-center">
                    {imageWidth} × {imageHeight} px
                  </p>
                  {isDicom && (
                    <p className="text-xs text-primary text-center font-medium">
                      DICOM {isVolume ? `Volume (${totalSlices} slices)` : 'Image'}
                    </p>
                  )}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Volume Slice Navigator Card */}
          {isVolume && totalSlices > 1 && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base flex items-center gap-2">
                  <Layers className="w-4 h-4" />
                  Volume Slices
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">Current Slice:</span>
                  <span className="font-medium text-primary">
                    {currentSlice + 1} / {totalSlices}
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max={totalSlices - 1}
                  value={currentSlice}
                  onChange={(e) => handleSliceChange(parseInt(e.target.value))}
                  disabled={isLoading}
                  className="w-full h-2 bg-border rounded-lg appearance-none cursor-pointer accent-primary disabled:opacity-50 disabled:cursor-not-allowed"
                />
                <div className="flex gap-2">
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => handleSliceChange(Math.max(0, currentSlice - 1))}
                    disabled={currentSlice === 0 || isLoading}
                    className="flex-1"
                  >
                    <ChevronLeft className="w-4 h-4 mr-1" />
                    Previous
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => handleSliceChange(Math.min(totalSlices - 1, currentSlice + 1))}
                    disabled={currentSlice === totalSlices - 1 || isLoading}
                    className="flex-1"
                  >
                    Next
                    <ChevronRight className="w-4 h-4 ml-1" />
                  </Button>
                </div>
                {/* 3D Propagation */}
                <Button
                  size="sm"
                  variant="default"
                  onClick={handlePropagate}
                  disabled={isLoading || !result?.masks?.length}
                  className="w-full"
                >
                  <Layers className="w-4 h-4 mr-2" />
                  Propagate to Volume
                </Button>
                {propagatedSlices.length > 0 && (
                  <p className="text-xs text-muted-foreground text-center">
                    Masks on {propagatedSlices.length} / {totalSlices} slices
                  </p>
                )}
              </CardContent>
            </Card>
          )}

          {/* Medical Modality Card */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center gap-2">
                <Stethoscope className="w-4 h-4" />
                Medical Modality
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <select
                value={selectedModality}
                onChange={(e) => handleModalityChange(e.target.value)}
                disabled={isLoading}
                className="w-full p-2 border border-border rounded-md bg-background text-sm"
              >
                {modalities.map((mod) => (
                  <option key={mod} value={mod}>
                    {mod.toUpperCase()}
                  </option>
                ))}
              </select>
              {modalityConfig && (
                <div className="text-xs text-muted-foreground space-y-1">
                  <div className="flex justify-between">
                    <span>Confidence:</span>
                    <span className="font-mono">{modalityConfig.confidence_threshold}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>NMS:</span>
                    <span className="font-mono">{modalityConfig.nms_threshold}</span>
                  </div>
                </div>
              )}
              {suggestions.length > 0 && showSuggestions && (
                <div className="space-y-2">
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <Lightbulb className="w-3 h-3" />
                    <span>Suggested prompts:</span>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {suggestions.slice(0, 6).map((suggestion) => (
                      <button
                        key={suggestion}
                        onClick={() => handleSuggestionClick(suggestion)}
                        className="px-2 py-1 text-xs bg-primary/10 hover:bg-primary/20 text-primary rounded-full transition-colors"
                      >
                        {suggestion}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Text Prompt Card */}
          <Card className={!sessionId ? "opacity-50 pointer-events-none" : ""}>
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center gap-2">
                <Type className="w-4 h-4" />
                Text Prompt
              </CardTitle>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleTextSubmit} className="flex gap-2">
                <Input
                  value={textPrompt}
                  onChange={(e) => setTextPrompt(e.target.value)}
                  placeholder='e.g. "person", "dog"'
                  disabled={isLoading}
                />
                <Button
                  type="submit"
                  disabled={isLoading || !textPrompt.trim()}
                >
                  {isLoading ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Sparkles className="w-4 h-4" />
                  )}
                </Button>
              </form>
            </CardContent>
          </Card>

          {/* Box Prompt Card */}
          <Card className={!sessionId ? "opacity-50 pointer-events-none" : ""}>
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center gap-2">
                <BoxSelect className="w-4 h-4" />
                Box Prompts
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <p className="text-xs text-muted-foreground">
                Draw boxes on the image to include or exclude regions
              </p>
              <div className="flex gap-2">
                <Button
                  variant={
                    interactionMode === "box" && boxMode === "positive"
                      ? "default"
                      : "secondary"
                  }
                  size="sm"
                  onClick={() => {
                    setInteractionMode("box");
                    setBoxMode("positive");
                  }}
                  className="flex-1"
                >
                  <Square className="w-4 h-4 mr-1" />
                  Include
                </Button>
                <Button
                  variant={
                    interactionMode === "box" && boxMode === "negative"
                      ? "destructive"
                      : "secondary"
                  }
                  size="sm"
                  onClick={() => {
                    setInteractionMode("box");
                    setBoxMode("negative");
                  }}
                  className="flex-1"
                >
                  <SquareMinus className="w-4 h-4 mr-1" />
                  Exclude
                </Button>
              </div>
              {interactionMode === "box" && (
                <div className="flex items-center gap-2 text-xs">
                  <div
                    className={`w-3 h-3 rounded border-2 ${
                      boxMode === "positive"
                        ? "border-primary bg-primary/20"
                        : "border-destructive bg-destructive/20"
                    }`}
                  />
                  <span className="text-muted-foreground">
                    Drawing: {boxMode === "positive" ? "Include" : "Exclude"}
                  </span>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Point Prompt Card */}
          <Card className={!sessionId ? "opacity-50 pointer-events-none" : ""}>
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center gap-2">
                <MousePointer2 className="w-4 h-4" />
                Point Prompts
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <p className="text-xs text-muted-foreground">
                Click points on the image to include or exclude regions
              </p>
              <div className="flex gap-2">
                <Button
                  variant={
                    interactionMode === "point" && pointMode === "positive"
                      ? "default"
                      : "secondary"
                  }
                  size="sm"
                  onClick={() => {
                    setInteractionMode("point");
                    setPointMode("positive");
                  }}
                  className="flex-1"
                >
                  <MousePointerClick className="w-4 h-4 mr-1" />
                  Include
                </Button>
                <Button
                  variant={
                    interactionMode === "point" && pointMode === "negative"
                      ? "destructive"
                      : "secondary"
                  }
                  size="sm"
                  onClick={() => {
                    setInteractionMode("point");
                    setPointMode("negative");
                  }}
                  className="flex-1"
                >
                  <MousePointerClick className="w-4 h-4 mr-1" />
                  Exclude
                </Button>
              </div>
              {interactionMode === "point" && (
                <div className="flex items-center gap-2 text-xs">
                  <div
                    className={`w-3 h-3 rounded-full border-2 ${
                      pointMode === "positive"
                        ? "border-primary bg-primary/20"
                        : "border-destructive bg-destructive/20"
                    }`}
                  />
                  <span className="text-muted-foreground">
                    Clicking: {pointMode === "positive" ? "Include" : "Exclude"}
                  </span>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Results & Actions */}
          <Card className={!sessionId ? "opacity-50 pointer-events-none" : ""}>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Results</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Objects found:</span>
                <span className="font-medium text-primary">{maskCount}</span>
              </div>
              {result?.prompted_boxes && result.prompted_boxes.length > 0 && (
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">Box prompts:</span>
                  <span className="font-medium">
                    {result.prompted_boxes.length}
                  </span>
                </div>
              )}
              {result?.prompted_points && result.prompted_points.length > 0 && (
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">Point prompts:</span>
                  <span className="font-medium">
                    {result.prompted_points.length}
                  </span>
                </div>
              )}
              <Button
                variant="destructive"
                size="sm"
                onClick={handleReset}
                disabled={isLoading}
                className="w-full"
              >
                <Trash2 className="w-4 h-4 mr-2" />
                Clear All Prompts
              </Button>
            </CardContent>
          </Card>

          {/* Performance Card */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center justify-between">
                <span className="flex items-center gap-2">
                  <Timer className="w-4 h-4" />
                  Performance
                </span>
                <span className="flex items-center gap-1 text-xs font-normal text-muted-foreground">
                  <Server className="w-3 h-3" />
                  server
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {/* Last request timing with highlight */}
              {lastTiming && (
                <div className="bg-primary/10 border border-primary/20 rounded-lg p-3 animate-fade-in-up">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-muted-foreground truncate max-w-[180px]">
                      {lastTiming.label}
                    </span>
                    <span className="text-sm font-bold text-primary">
                      {formatDuration(lastTiming.duration)}
                    </span>
                  </div>
                </div>
              )}

              {/* Stats summary */}
              {timings.length > 0 && (
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="bg-card border border-border rounded p-2">
                    <div className="text-muted-foreground">Requests</div>
                    <div className="font-medium">{timings.length}</div>
                  </div>
                  {avgInferenceTime !== null && (
                    <div className="bg-card border border-border rounded p-2">
                      <div className="text-muted-foreground">Avg Inference</div>
                      <div className="font-medium">
                        {formatDuration(avgInferenceTime)}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Recent requests log */}
              {timings.length > 0 && (
                <div className="space-y-1 max-h-32 overflow-y-auto">
                  <div className="text-xs text-muted-foreground mb-1">
                    Recent:
                  </div>
                  {timings.slice(0, 5).map((timing, idx) => (
                    <div
                      key={idx}
                      className="flex items-center justify-between text-xs py-1 border-b border-border/50 last:border-0"
                    >
                      <span className="text-muted-foreground truncate max-w-[160px]">
                        {timing.label}
                      </span>
                      <span className="font-mono text-foreground">
                        {formatDuration(timing.duration)}
                      </span>
                    </div>
                  ))}
                </div>
              )}

              {timings.length === 0 && (
                <p className="text-xs text-muted-foreground text-center py-2">
                  No requests yet
                </p>
              )}
            </CardContent>
          </Card>

          {/* Error Display */}
          {error && (
            <Card className="border-destructive bg-destructive/10">
              <CardContent className="py-3">
                <p className="text-sm text-destructive">{error}</p>
              </CardContent>
            </Card>
          )}
        </aside>

        {/* Main Canvas Area */}
        <section>
          <Card className="overflow-hidden">
            <CardContent className="p-4">
              <SegmentationCanvas
                imageUrl={imageUrl}
                imageWidth={imageWidth}
                imageHeight={imageHeight}
                result={result}
                boxMode={boxMode}
                pointMode={pointMode}
                interactionMode={interactionMode}
                onBoxDrawn={handleBoxDrawn}
                onPointClicked={handlePointClicked}
                isLoading={isLoading}
              />
            </CardContent>
          </Card>

          {/* Keyboard Shortcuts */}
          {sessionId && (
            <div className="mt-4 flex flex-wrap gap-4 text-xs text-muted-foreground animate-fade-in-up">
              <div className="flex items-center gap-2">
                <kbd className="px-2 py-1 bg-card rounded border border-border font-mono">
                  Click + Drag
                </kbd>
                <span>Draw box</span>
              </div>
              <div className="flex items-center gap-2">
                <kbd className="px-2 py-1 bg-card rounded border border-border font-mono">
                  Click
                </kbd>
                <span>Add point</span>
              </div>
              <div className="flex items-center gap-2">
                <kbd className="px-2 py-1 bg-card rounded border border-border font-mono">
                  Enter
                </kbd>
                <span>Submit text prompt</span>
              </div>
            </div>
          )}
        </section>
      </div>

      {/* Footer */}
      <footer className="max-w-7xl mx-auto mt-12 pt-6 border-t border-border">
        <p className="text-xs text-muted-foreground text-center">
          MedSAM3 Medical Image Segmentation • 10+ Modalities • LoRA Fine-Tuning • MLX Backend • Next.js Frontend
        </p>
      </footer>
    </main>
  );
}
