package main

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/hybrid-gpu-scheduler/internal/executor"
	"github.com/hybrid-gpu-scheduler/internal/gpumonitor"
	"github.com/hybrid-gpu-scheduler/internal/metrics"
	"github.com/hybrid-gpu-scheduler/internal/scheduler"
	"github.com/hybrid-gpu-scheduler/pkg/gpu-devices"
	"github.com/hybrid-gpu-scheduler/pkg/llmscheduler"
	"github.com/hybrid-gpu-scheduler/pkg/types"
	"github.com/hybrid-gpu-scheduler/pkg/vram"
	"github.com/hybrid-gpu-scheduler/pkg/websocket"
	hgpub "github.com/hybrid-gpu-scheduler/pkg/backend"
	"github.com/gin-gonic/gin"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

func main() {
	log.Println("Hybrid GPU Scheduler starting...")
	log.Println("Unified NVIDIA + AMD GPU Scheduler v3.0 (Training + LLM Inference)")

	sched := scheduler.NewScheduler()

	// 发现真实 GPU，替换默认的模拟数据
	log.Println("[main] Discovering real GPUs...")
	DiscoverAndRegisterRealGPUs(sched)

	execInst := executor.GetExecutor()

	execInst.SetSSELogger(websocket.NewLogBridge())

	execInst.SetOnTaskDone(func(taskID string) {
		sched.ReleaseTask(taskID)
	})

	// GPU 监控
	gpuMon := gpumonitor.NewGPUMonitor(func(gpuID string, update gpumonitor.GPUMonitorUpdate) {
		usage := types.GPUUsage{
			UsedVRAMMB:   update.UsedVRAMMB,
			Utilization:  update.UtilizationPct,
			TemperatureC: update.TemperatureC,
			PowerDrawW:  update.PowerDrawW,
		}
		sched.UpdateGPUUsage(gpuID, usage)
	})
	go gpuMon.Start(5 * time.Second)
	defer gpuMon.Stop()

	gpus := sched.ListGPUs()
	log.Printf("Scheduler initialized with %d GPUs", len(gpus))
	for _, gpu := range gpus {
		log.Printf("  GPU %s: %s (%d MB)", gpu.Info.ID, gpu.Info.Name, gpu.Info.VRAMMB)
	}

	// ============ LLM 后端初始化 ============
	llmPlanner := vram.NewPlanner()
	llmBackendMgr := hgpub.NewBackendManager()

	// 注册 Ollama 后端（默认 localhost:11434）
	if ollamaURL := os.Getenv("OLLAMA_URL"); ollamaURL != "" {
		ollama := hgpub.NewOllamaBackend(ollamaURL)
		llmBackendMgr.Register("ollama", ollama)
		log.Printf("LLM backend: Ollama registered at %s", ollamaURL)
	} else {
		ollama := hgpub.NewOllamaBackend("http://localhost:11434")
		llmBackendMgr.Register("ollama", ollama)
		log.Printf("LLM backend: Ollama registered at http://localhost:11434 (default)")
	}

	// 注册 LM Studio 后端（可选）
	if lmstudioURL := os.Getenv("LMSTUDIO_URL"); lmstudioURL != "" {
		lmstudio := hgpub.NewLMStudioBackend(lmstudioURL)
		llmBackendMgr.Register("lmstudio", lmstudio)
		log.Printf("LLM backend: LM Studio registered at %s", lmstudioURL)
	}

	// 注册 vLLM 后端（可选，需要 API_KEY）
	if vllmURL := os.Getenv("VLLM_URL"); vllmURL != "" {
		apiKey := os.Getenv("VLLM_API_KEY")
		vllm := hgpub.NewVLLMBackend(vllmURL, apiKey)
		llmBackendMgr.Register("vllm", vllm)
		log.Printf("LLM backend: vLLM registered at %s", vllmURL)
	}

	// 注册 GPU 到 VRAM Planner
	for _, gpu := range gpus {
		maxModelMB := int(float64(gpu.Info.VRAMMB) * 0.85)
		llmPlanner.RegisterGPU(gpu.Info.ID, string(gpu.Info.Type), int(gpu.Info.VRAMMB), maxModelMB)
	}

	// 初始化 LLM 调度器
	llmSched := llmscheduler.NewScheduler(llmscheduler.Config{
		QueueSize:      100,
		BatchTimeoutMs: 200,
		MaxBatchSize:   8,
	}, llmPlanner)
	llmSched.Start()
	llmscheduler.SetBackendManager(llmBackendMgr)
	log.Println("LLM Scheduler started")

	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()

	r.GET("/metrics", gin.WrapH(promhttp.Handler()))

	r.GET("/ws", func(c *gin.Context) {
		websocket.GetManager().HandleWebSocket(c.Writer, c.Request)
	})

	r.Use(httpMetricsMiddleware())
	r.Use(gin.Logger())
	r.Use(gin.Recovery())

	// Dashboard
	exePath, _ := os.Executable()
	exeDir := filepath.Dir(exePath)
	dash := filepath.Join(exeDir, "dashboard.html")
	if _, err := os.Stat(dash); os.IsNotExist(err) {
		log.Printf("WARNING: dashboard.html not found at %s", dash)
	}
	dash520 := filepath.Join(exeDir, "dashboard-520.html")

	r.GET("/dashboard", func(c *gin.Context) {
		c.Header("Content-Type", "text/html; charset=utf-8")
		c.File(dash)
	})
	r.GET("/dashboard-520", func(c *gin.Context) {
		c.Header("Content-Type", "text/html; charset=utf-8")
		c.File(dash520)
	})
	r.GET("/", func(c *gin.Context) {
		c.Header("Content-Type", "text/html; charset=utf-8")
		c.File(dash)
	})

	// Health
	r.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status":  "ok",
			"service": "hybrid-gpu-scheduler",
			"version": "3.0-unified",
			"gpus":    len(gpus),
		})
	})

	// ===== Tasks API (训练/通用任务) =====
	tasks := r.Group("/api/tasks")
	{
		tasks.POST("", func(c *gin.Context) {
			body, err := io.ReadAll(c.Request.Body)
			if err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": "failed to read body"})
				return
			}
			req, err := types.ParseTaskRequest(body)
			if err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}

			if req.Policy != "" {
				sched.SetDefaultPolicy(types.SchedulingPolicy(req.Policy))
			}

			task := &types.Task{
				Name:     req.Name,
				Type:     req.Type,
				Priority: req.Priority,
				GPUReq:   req.GPUReq,
				VRAMMB:   req.VRAMMB,
				Command:  req.Command,
				Args:     req.Args,
				Env:      req.Env,
				WorkDir:  req.WorkDir,
				Timeout:  req.Timeout,
			}

			metrics.RecordTaskSubmit(req.Type, req.Priority)
			assignedGPU, err := sched.SubmitTask(task)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			var createdTask *types.Task
			for _, t := range sched.ListTasks() {
				if t.ID == task.ID {
					createdTask = t
					break
				}
			}

			result := "success"
			if assignedGPU == "" {
				result = "no_gpu"
			}
			metrics.RecordScheduling(string(sched.GetDefaultPolicy()), result, 0)

			c.JSON(http.StatusOK, gin.H{
				"success": true,
				"task_id": task.ID,
				"task":    createdTask,
				"policy":  sched.GetDefaultPolicy(),
			})
		})

		tasks.GET("", func(c *gin.Context) {
			t := sched.ListTasks()
			c.JSON(http.StatusOK, gin.H{"success": true, "tasks": t})
		})

		tasks.GET("/:id", func(c *gin.Context) {
			taskID := c.Param("id")
			for _, t := range sched.ListTasks() {
				if t.ID == taskID {
					c.JSON(http.StatusOK, gin.H{"success": true, "task": t})
					return
				}
			}
			c.JSON(http.StatusNotFound, gin.H{"success": false, "error": "Task not found"})
		})

		tasks.DELETE("/:id", func(c *gin.Context) {
			taskID := c.Param("id")
			if sched.CancelTask(taskID) {
				c.JSON(http.StatusOK, gin.H{"success": true, "message": "Task cancelled"})
			} else {
				c.JSON(http.StatusNotFound, gin.H{"success": false, "error": "Task not found"})
			}
		})

		tasks.POST("/:id/complete", func(c *gin.Context) {
			taskID := c.Param("id")
			if sched.CompleteTask(taskID) {
				c.JSON(http.StatusOK, gin.H{"success": true, "message": "Task completed"})
			} else {
				c.JSON(http.StatusNotFound, gin.H{"success": false, "error": "Task not found"})
			}
		})

		tasks.POST("/:id/stop", func(c *gin.Context) {
			taskID := c.Param("id")
			found := false
			for _, t := range sched.ListTasks() {
				if t.ID == taskID {
					found = true
					break
				}
			}
			if !found {
				c.JSON(http.StatusNotFound, gin.H{"success": false, "error": "Task not found"})
				return
			}
			if err := execInst.StopTask(taskID); err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"success": false, "error": err.Error()})
				return
			}
			c.JSON(http.StatusOK, gin.H{"success": true, "message": "Task stop requested", "task_id": taskID})
		})

		tasks.GET("/:id/log", func(c *gin.Context) {
			taskID := c.Param("id")
			var logContent string
			for _, t := range sched.ListTasks() {
				if t.ID == taskID {
					if t.LogFile != "" {
						data, err := os.ReadFile(t.LogFile)
						if err == nil {
							logContent = string(data)
						}
					}
					break
				}
			}
			if logContent == "" {
				logContent = "No log available"
			}
			c.JSON(http.StatusOK, gin.H{"success": true, "task_id": taskID, "log": logContent})
		})
	}

	// ===== GPU API =====
	gpusAPI := r.Group("/api/gpus")
	{
		gpusAPI.GET("", func(c *gin.Context) {
			g := sched.ListGPUs()
			for _, gpu := range g {
				metrics.RecordGPUUpdate(gpu.Info.ID, string(gpu.Info.Type), gpu.Info.Name,
					gpu.Usage.UsedVRAMMB, gpu.Info.VRAMMB, gpu.Usage.Utilization, gpu.Usage.RunningTasks)
				metrics.GPUScore.WithLabelValues(gpu.Info.ID, string(gpu.Info.Type), gpu.Info.Name).Set(gpu.Score)
			}
			c.JSON(http.StatusOK, gin.H{"success": true, "gpus": g})
		})

		gpusAPI.GET("/:id", func(c *gin.Context) {
			gpuID := c.Param("id")
			gpu, ok := sched.GetGPU(gpuID)
			if !ok {
				c.JSON(http.StatusNotFound, gin.H{"success": false, "error": "GPU not found"})
				return
			}
			metrics.RecordGPUUpdate(gpu.Info.ID, string(gpu.Info.Type), gpu.Info.Name,
				gpu.Usage.UsedVRAMMB, gpu.Info.VRAMMB, gpu.Usage.Utilization, gpu.Usage.RunningTasks)
			metrics.GPUScore.WithLabelValues(gpu.Info.ID, string(gpu.Info.Type), gpu.Info.Name).Set(gpu.Score)
			c.JSON(http.StatusOK, gin.H{"success": true, "gpu": gpu})
		})

		gpusAPI.GET("/:id/score", func(c *gin.Context) {
			gpuID := c.Param("id")
			taskType := c.DefaultQuery("task_type", "training")
			vramMB := 4096
			fmt.Sscanf(c.DefaultQuery("vram_mb", "4096"), "%d", &vramMB)
			scoreBreakdown := sched.GetScoreBreakdown(gpuID, taskType, vramMB)
			c.JSON(http.StatusOK, gin.H{
				"success":         true,
				"gpu_id":          gpuID,
				"task_type":       taskType,
				"vram_required":   vramMB,
				"score_breakdown": scoreBreakdown,
			})
		})
	}

	// ===== Scheduler API =====
	schedAPI := r.Group("/api/scheduler")
	{
		schedAPI.GET("/policy", func(c *gin.Context) {
			policy := sched.GetDefaultPolicy()
			c.JSON(http.StatusOK, gin.H{
				"success":     true,
				"policy":      policy,
				"description": getPolicyDescription(policy),
			})
		})

		schedAPI.POST("/policy", func(c *gin.Context) {
			var req struct{ Policy string `json:"policy"` }
			if err := c.ShouldBindJSON(&req); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}
			sched.SetDefaultPolicy(types.SchedulingPolicy(req.Policy))
			c.JSON(http.StatusOK, gin.H{
				"success":     true,
				"policy":      req.Policy,
				"description": getPolicyDescription(types.SchedulingPolicy(req.Policy)),
			})
		})

		schedAPI.GET("/status", func(c *gin.Context) {
			tasks := sched.ListTasks()
			gpus := sched.ListGPUs()
			pendingCount, runningCount := 0, 0
			for _, t := range tasks {
				if t.Status == "pending" {
					pendingCount++
				} else if t.Status == "running" {
					runningCount++
				}
			}
			c.JSON(http.StatusOK, gin.H{
				"success":       true,
				"total_tasks":   len(tasks),
				"pending_tasks": pendingCount,
				"running_tasks": runningCount,
				"total_gpus":    len(gpus),
				"policy":        sched.GetDefaultPolicy(),
			})
		})
	}

	// ===== Executor API =====
	execAPI := r.Group("/api/executor")
	{
		execAPI.GET("/running", func(c *gin.Context) {
			running := execInst.ListRunningTasks()
			ids := make([]string, 0, len(running))
			for _, rt := range running {
				ids = append(ids, rt.Task.ID+" (PID:"+strconv.Itoa(rt.PID)+")")
			}
			c.JSON(http.StatusOK, gin.H{"success": true, "tasks": ids, "count": len(ids)})
		})
		execAPI.GET("/status", func(c *gin.Context) {
			c.JSON(http.StatusOK, gin.H{"success": true, "status": "running", "active_tasks": len(execInst.ListRunningTasks())})
		})
	}

	// Preemption API
	SetupPreemptAPI(r, sched)

	// ===== LLM Inference API (新增 v3.0) =====
	llmAPI := r.Group("/api/llm")
	{
		// 列出所有后端
		llmAPI.GET("/backends", func(c *gin.Context) {
			allModels, _ := llmBackendMgr.ListAllModels()
			backends := make([]map[string]interface{}, 0)
			for _, name := range llmBackendMgr.Order() {
				b := llmBackendMgr.Get(name)
				if b != nil {
					models := allModels[name]
					backends = append(backends, map[string]interface{}{
						"name":   name,
						"models": models,
					})
				}
			}
			c.JSON(http.StatusOK, gin.H{"success": true, "backends": backends})
		})

		// 列出所有模型
		llmAPI.GET("/models", func(c *gin.Context) {
			allModels, err := llmBackendMgr.ListAllModels()
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}
			c.JSON(http.StatusOK, gin.H{"success": true, "models": allModels})
		})

		// 加载模型
		llmAPI.POST("/models/load", func(c *gin.Context) {
			var req struct {
				Model   string `json:"model" binding:"required"`
				Backend string `json:"backend"`
			}
			if err := c.ShouldBindJSON(&req); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}

			var b hgpub.Backend
			if req.Backend != "" {
				b = llmBackendMgr.Get(req.Backend)
			} else {
				b = llmBackendMgr.Primary()
			}
			if b == nil {
				c.JSON(http.StatusNotFound, gin.H{"error": "Backend not found"})
				return
			}

			if err := b.LoadModel(req.Model); err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}
			c.JSON(http.StatusOK, gin.H{"success": true, "message": fmt.Sprintf("Model %s loaded on %s", req.Model, b.Name())})
		})

		// 卸载模型
		llmAPI.POST("/models/unload", func(c *gin.Context) {
			var req struct {
				Model   string `json:"model" binding:"required"`
				Backend string `json:"backend"`
			}
			if err := c.ShouldBindJSON(&req); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}

			var b hgpub.Backend
			if req.Backend != "" {
				b = llmBackendMgr.Get(req.Backend)
			} else {
				b = llmBackendMgr.Primary()
			}
			if b == nil {
				c.JSON(http.StatusNotFound, gin.H{"error": "Backend not found"})
				return
			}

			if err := b.UnloadModel(req.Model); err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}
			c.JSON(http.StatusOK, gin.H{"success": true, "message": fmt.Sprintf("Model %s unloaded from %s", req.Model, b.Name())})
		})

		// 聊天推理（非流式）
		llmAPI.POST("/chat", func(c *gin.Context) {
			var req struct {
				Model      string                 `json:"model" binding:"required"`
				Messages   []hgpub.ChatMessage    `json:"messages" binding:"required"`
				Stream     bool                   `json:"stream"`
				Backend    string                 `json:"backend"`
				Options    map[string]interface{} `json:"options"`
			}
			if err := c.ShouldBindJSON(&req); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}

			var b hgpub.Backend
			if req.Backend != "" {
				b = llmBackendMgr.Get(req.Backend)
			} else {
				b = llmBackendMgr.Primary()
			}
			if b == nil {
				c.JSON(http.StatusNotFound, gin.H{"error": "No backend available"})
				return
			}

			chatReq := hgpub.ChatRequest{
				Model:    req.Model,
				Messages: req.Messages,
				Stream:   false,
				Options:  req.Options,
			}

			resp, err := b.Chat(chatReq)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, gin.H{
				"success": true,
				"backend": b.Name(),
				"model":   resp.Model,
				"reply":   resp.Message.Content,
				"done":    resp.Done,
			})
		})

		// 流式推理
		llmAPI.POST("/chat/stream", func(c *gin.Context) {
			var req struct {
				Model    string              `json:"model" binding:"required"`
				Messages []hgpub.ChatMessage `json:"messages" binding:"required"`
				Backend  string              `json:"backend"`
			}
			if err := c.ShouldBindJSON(&req); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}

			var b hgpub.Backend
			if req.Backend != "" {
				b = llmBackendMgr.Get(req.Backend)
			} else {
				b = llmBackendMgr.Primary()
			}
			if b == nil {
				c.JSON(http.StatusNotFound, gin.H{"error": "No backend available"})
				return
			}

			chatReq := hgpub.ChatRequest{
				Model:    req.Model,
				Messages: req.Messages,
				Stream:   true,
			}

			stream, err := b.ChatStream(chatReq)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}
			defer stream.Close()

			c.Header("Content-Type", "text/event-stream")
			c.Header("Cache-Control", "no-cache")
			c.Header("Connection", "keep-alive")
			c.Stream(func(w io.Writer) bool {
				buf := make([]byte, 4096)
				n, err := stream.Read(buf)
				if err != nil {
					return false
				}
				io.WriteString(w, string(buf[:n]))
				return true
			})
		})

		// VRAM 规划：估算模型显存需求
		llmAPI.GET("/vram/estimate", func(c *gin.Context) {
			model := c.Query("model")
			if model == "" {
				c.JSON(http.StatusBadRequest, gin.H{"error": "model parameter required"})
				return
			}
			vramMB := llmPlanner.EstimateModelVRAM(model)
			category := llmPlanner.GetModelCategory(model)
			gpuStatus := llmPlanner.GetGPUStatus()
			c.JSON(http.StatusOK, gin.H{
				"success":       true,
				"model":         model,
				"vram_mb":       vramMB,
				"category":      category,
				"gpu_capacity":  gpuStatus,
			})
		})

		// 状态
		llmAPI.GET("/status", func(c *gin.Context) {
			status := llmSched.GetStatus()
			allModels, _ := llmBackendMgr.ListAllModels()
			c.JSON(http.StatusOK, gin.H{
				"success": true,
				"scheduler": status,
				"backends": allModels,
			})
		})
	}

	addr := ":8080"
	log.Printf("")
	log.Printf("API server listening on %s", addr)
	log.Printf("  Health:    http://localhost%s/health", addr)
	log.Printf("  Metrics:   http://localhost%s/metrics  (Prometheus)", addr)
	log.Printf("  Dashboard: http://localhost%s/dashboard", addr)
	log.Printf("  Tasks:     http://localhost%s/api/tasks     (训练/通用任务)", addr)
	log.Printf("  GPUs:      http://localhost%s/api/gpus", addr)
	log.Printf("  LLM:       http://localhost%s/api/llm/chat  (LLM 推理)", addr)
	log.Printf("")

	if err := r.Run(addr); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}

func httpMetricsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		path := c.FullPath()
		if path == "" {
			path = c.Request.URL.Path
		}
		c.Next()
		status := strconv.Itoa(c.Writer.Status())
		duration := time.Since(start).Seconds()
		metrics.HTTPRequestsTotal.WithLabelValues(c.Request.Method, path, status).Inc()
		metrics.HTTPRequestDuration.WithLabelValues(c.Request.Method, path).Observe(duration)
	}
}

func getPolicyDescription(policy types.SchedulingPolicy) string {
	switch policy {
	case types.PolicyBinpack:
		return "Binpack: compact scheduling"
	case types.PolicySpread:
		return "Spread: load balancing"
	case types.PolicyGPUType:
		return "GPU Type: type-first scheduling"
	default:
		return "Unknown policy"
	}
}

// DiscoverAndRegisterRealGPUs discovers real GPUs via nvidia-smi / rocm-smi
// and registers them with the scheduler (replacing default simulated ones).
func DiscoverAndRegisterRealGPUs(sched *scheduler.Scheduler) {
	mgr := gpudevices.GetGPUDeviceManager()
	err := mgr.Refresh()
	if err != nil {
		log.Printf("[main] Warning: could not discover real GPUs: %v", err)
		log.Println("[main] Falling back to default (simulated) GPU configuration")
		return
	}

	gpus, err := mgr.GetAllGPUs()
	if err != nil {
		log.Printf("[main] Warning: failed to get GPU info: %v", err)
		log.Println("[main] Falling back to default (simulated) GPU configuration")
		return
	}

	if len(gpus) == 0 {
		log.Println("[main] Warning: no GPUs discovered, keeping default configuration")
		return
	}

	// 用真实 GPU 替换默认的模拟 GPU
	for _, g := range gpus {
		gpuType := types.GPUTypeNVIDIA
		if g.Vendor == "AMD" {
			gpuType = types.GPUTypeAMD
		}

		snap := &types.GPUSnapshot{
			Info: types.GPUInfo{
				ID:           strconv.Itoa(g.Index),
				Type:         gpuType,
				Name:         g.Name,
				VRAMMB:       int(g.MemoryTotal), // MB
				CoreCount:    0, //  unknown from smi
				ComputeUnits: 0, //  unknown from smi
				Architecture: "", //  unknown from smi
			},
			Usage: types.GPUUsage{
				ID:           strconv.Itoa(g.Index),
				UsedVRAMMB:   int(g.MemoryUsed),
				Utilization:  g.Utilization,
				MemoryUtil:   float64(g.MemoryUsed) * 100.0 / float64(g.MemoryTotal),
				ComputeUtil:  g.Utilization,
				RunningTasks: 0,
				LastUpdated:  time.Now(),
			},
		}

		sched.RegisterGPU(snap)
		log.Printf("[main] Registered real GPU: %s (%s, %d MB VRAM)", strconv.Itoa(g.Index), g.Name, g.MemoryTotal)
	}

	log.Printf("[main] Total real GPUs registered: %d", len(gpus))
}
