package opencode

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/chenhg5/cc-connect/core"
)

func init() {
	core.RegisterAgent("opencode", New)
}

// Agent drives the OpenCode CLI in headless mode using `opencode run --format json`.
//
// Modes:
//   - "default": standard mode
//   - "yolo":    auto mode (opencode run is auto by default in non-interactive mode)
type Agent struct {
	workDir    string
	model      string
	mode       string
	cmd        string // CLI binary name, default "opencode"
	providers  []core.ProviderConfig
	activeIdx  int
	sessionEnv []string
	mu         sync.RWMutex
}

func New(opts map[string]any) (core.Agent, error) {
	workDir, _ := opts["work_dir"].(string)
	if workDir == "" {
		workDir = "."
	}
	model, _ := opts["model"].(string)
	mode, _ := opts["mode"].(string)
	mode = normalizeMode(mode)
	cmd, _ := opts["cmd"].(string)
	if cmd == "" {
		cmd = "opencode"
	}

	if _, err := exec.LookPath(cmd); err != nil {
		return nil, fmt.Errorf("opencode: %q CLI not found in PATH, install from: https://github.com/opencode-ai/opencode", cmd)
	}

	return &Agent{
		workDir:   workDir,
		model:     model,
		mode:      mode,
		cmd:       cmd,
		activeIdx: -1,
	}, nil
}

func normalizeMode(raw string) string {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "yolo", "auto", "force", "bypasspermissions":
		return "yolo"
	default:
		return "default"
	}
}

func (a *Agent) Name() string { return "opencode" }

func (a *Agent) SetWorkDir(dir string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.workDir = dir
	slog.Info("opencode: work_dir changed", "work_dir", dir)
}

func (a *Agent) GetWorkDir() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.workDir
}

func (a *Agent) SetModel(model string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.model = model
	slog.Info("opencode: model changed", "model", model)
}

func (a *Agent) GetModel() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return core.GetProviderModel(a.providers, a.activeIdx, a.model)
}

func (a *Agent) configuredModels() []core.ModelOption {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return core.GetProviderModels(a.providers, a.activeIdx)
}

func (a *Agent) AvailableModels(_ context.Context) []core.ModelOption {
	if models := a.configuredModels(); len(models) > 0 {
		return models
	}
	return []core.ModelOption{
		{Name: "anthropic/claude-sonnet-4-20250514", Desc: "Claude Sonnet 4 (default)"},
		{Name: "anthropic/claude-opus-4-20250514", Desc: "Claude Opus 4"},
		{Name: "openai/gpt-4o", Desc: "GPT-4o"},
		{Name: "openai/o3", Desc: "OpenAI o3"},
	}
}

func (a *Agent) SetSessionEnv(env []string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.sessionEnv = env
}

func (a *Agent) StartSession(ctx context.Context, sessionID string) (core.AgentSession, error) {
	a.mu.Lock()
	model := a.model
	mode := a.mode
	cmd := a.cmd
	workDir := a.workDir
	extraEnv := a.providerEnvLocked()
	extraEnv = append(extraEnv, a.sessionEnv...)
	if a.activeIdx >= 0 && a.activeIdx < len(a.providers) {
		if m := a.providers[a.activeIdx].Model; m != "" {
			model = m
		}
	}
	a.mu.Unlock()

	return newOpencodeSession(ctx, cmd, workDir, model, mode, sessionID, extraEnv)
}

// ListSessions runs `opencode session list` and parses the JSON output.
func (a *Agent) ListSessions(_ context.Context) ([]core.AgentSessionInfo, error) {
	return listOpencodeSessions(a.cmd, a.workDir)
}

func (a *Agent) Stop() error { return nil }

// -- ModeSwitcher --

func (a *Agent) SetMode(mode string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.mode = normalizeMode(mode)
	slog.Info("opencode: mode changed", "mode", a.mode)
}

func (a *Agent) GetMode() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.mode
}

func (a *Agent) PermissionModes() []core.PermissionModeInfo {
	return []core.PermissionModeInfo{
		{Key: "default", Name: "Default", NameZh: "默认", Desc: "Standard mode", DescZh: "标准模式"},
		{Key: "yolo", Name: "YOLO", NameZh: "全自动", Desc: "Auto-approve all tool calls", DescZh: "自动批准所有工具调用"},
	}
}

// -- ContextCompressor --

func (a *Agent) CompressCommand() string { return "/compact" }

// -- MemoryFileProvider --

func (a *Agent) ProjectMemoryFile() string {
	absDir, err := filepath.Abs(a.workDir)
	if err != nil {
		absDir = a.workDir
	}
	return filepath.Join(absDir, "OPENCODE.md")
}

func (a *Agent) GlobalMemoryFile() string {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(homeDir, ".opencode", "OPENCODE.md")
}

// -- ProviderSwitcher --

func (a *Agent) SetProviders(providers []core.ProviderConfig) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.providers = providers
}

func (a *Agent) SetActiveProvider(name string) bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	if name == "" {
		a.activeIdx = -1
		slog.Info("opencode: provider cleared")
		return true
	}
	for i, p := range a.providers {
		if p.Name == name {
			a.activeIdx = i
			slog.Info("opencode: provider switched", "provider", name)
			return true
		}
	}
	return false
}

func (a *Agent) GetActiveProvider() *core.ProviderConfig {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.activeIdx < 0 || a.activeIdx >= len(a.providers) {
		return nil
	}
	p := a.providers[a.activeIdx]
	return &p
}

func (a *Agent) ListProviders() []core.ProviderConfig {
	a.mu.Lock()
	defer a.mu.Unlock()
	result := make([]core.ProviderConfig, len(a.providers))
	copy(result, a.providers)
	return result
}

func (a *Agent) providerEnvLocked() []string {
	if a.activeIdx < 0 || a.activeIdx >= len(a.providers) {
		return nil
	}
	p := a.providers[a.activeIdx]
	var env []string
	if p.APIKey != "" {
		env = append(env, "ANTHROPIC_API_KEY="+p.APIKey)
	}
	for k, v := range p.Env {
		env = append(env, k+"="+v)
	}
	return env
}

// -- HistoryProvider --

type opencodeExportData struct {
	Messages []opencodeExportMessage `json:"messages"`
}

type opencodeExportMessage struct {
	Info struct {
		Role string `json:"role"`
		Time struct {
			Created int64 `json:"created"`
		} `json:"time"`
	} `json:"info"`
	Parts []opencodeExportPart `json:"parts"`
}

type opencodeExportPart struct {
	Type  string `json:"type"`
	Text  string `json:"text,omitempty"`
	Tool  string `json:"tool,omitempty"`
	State *struct {
		Input  map[string]any `json:"input,omitempty"`
		Output string         `json:"output,omitempty"`
	} `json:"state,omitempty"`
}

func (a *Agent) GetSessionHistory(_ context.Context, sessionID string, limit int) ([]core.HistoryEntry, error) {
	c := exec.Command(a.cmd, "export", sessionID)
	c.Dir = a.GetWorkDir()

	out, err := c.Output()
	if err != nil {
		return nil, fmt.Errorf("opencode: export session: %w", err)
	}

	outStr := string(out)
	idx := strings.Index(outStr, "{")
	if idx < 0 {
		return nil, fmt.Errorf("opencode: no JSON found in export output")
	}

	var data opencodeExportData
	if err := json.Unmarshal([]byte(outStr[idx:]), &data); err != nil {
		return nil, fmt.Errorf("opencode: parse export JSON: %w", err)
	}

	var entries []core.HistoryEntry
	for _, msg := range data.Messages {
		var sb strings.Builder
		for _, p := range msg.Parts {
			switch p.Type {
			case "text":
				if p.Text != "" {
					sb.WriteString(p.Text)
				}
			case "tool":
				sb.WriteString(fmt.Sprintf("\n> 🔧 Tool: %s", p.Tool))
				if p.State != nil && p.State.Input != nil {
					if cmd, ok := p.State.Input["command"]; ok {
						sb.WriteString(fmt.Sprintf(" `%v`", cmd))
					} else if query, ok := p.State.Input["query"]; ok {
						sb.WriteString(fmt.Sprintf(" `%v`", query))
					} else if pattern, ok := p.State.Input["pattern"]; ok {
						sb.WriteString(fmt.Sprintf(" `%v`", pattern))
						if path, ok := p.State.Input["path"]; ok {
							sb.WriteString(fmt.Sprintf(" in %v", path))
						}
					}
				}
				sb.WriteString("\n")
			}
		}

		text := strings.TrimSpace(sb.String())
		if text == "" {
			continue
		}

		entries = append(entries, core.HistoryEntry{
			Role:      msg.Info.Role,
			Content:   text,
			Timestamp: time.UnixMilli(msg.Info.Time.Created),
		})
	}

	if limit > 0 && limit < len(entries) {
		entries = entries[len(entries)-limit:]
	}

	return entries, nil
}

// -- Session listing --

// opencodeSessionEntry represents a session from `opencode session list` output.
type opencodeSessionEntry struct {
	ID        string `json:"id"`
	Title     string `json:"title"`
	Updated   int64  `json:"updated"` // Unix timestamp in milliseconds
	Created   int64  `json:"created"`
	ProjectID string `json:"projectId"`
	Directory string `json:"directory"`
}

func listOpencodeSessions(cmd, workDir string) ([]core.AgentSessionInfo, error) {
	c := exec.Command(cmd, "session", "list", "--format", "json")
	c.Dir = workDir

	out, err := c.Output()
	if err != nil {
		return nil, fmt.Errorf("opencode: session list: %w", err)
	}

	var entries []opencodeSessionEntry
	if err := json.Unmarshal(out, &entries); err != nil {
		return nil, fmt.Errorf("opencode: parse session list: %w", err)
	}

	absWorkDir, err := filepath.Abs(workDir)
	if err != nil {
		absWorkDir = workDir
	}

	msgCounts := querySessionMessageCounts()

	var sessions []core.AgentSessionInfo
	for _, e := range entries {
		// Filter by directory if available
		if e.Directory != "" {
			absEntryDir, err := filepath.Abs(e.Directory)
			if err != nil {
				absEntryDir = e.Directory
			}
			if !strings.EqualFold(absEntryDir, absWorkDir) {
				continue
			}
		}

		sessions = append(sessions, core.AgentSessionInfo{
			ID:           e.ID,
			Summary:      e.Title,
			MessageCount: msgCounts[e.ID],
			ModifiedAt:   time.UnixMilli(e.Updated),
		})
	}

	return sessions, nil
}

// querySessionMessageCounts uses the sqlite3 CLI to read message counts from
// OpenCode's local database. Returns an empty map on any failure.
func querySessionMessageCounts() map[string]int {
	dbPath := opencodeDBPath()
	if dbPath == "" {
		return nil
	}
	if _, err := os.Stat(dbPath); err != nil {
		return nil
	}
	sqlite3, err := exec.LookPath("sqlite3")
	if err != nil {
		slog.Warn("opencode: sqlite3 CLI not found, message counts unavailable", "err", err)
		return nil
	}

	out, err := exec.Command(sqlite3, dbPath,
		"SELECT session_id, COUNT(*) FROM message GROUP BY session_id").Output()
	if err != nil {
		slog.Warn("opencode: sqlite3 query failed", "db_path", dbPath, "err", err)
		return nil
	}

	counts := make(map[string]int)
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		parts := strings.SplitN(line, "|", 2)
		if len(parts) != 2 {
			continue
		}
		var n int
		if _, err := fmt.Sscanf(parts[1], "%d", &n); err == nil {
			counts[parts[0]] = n
		}
	}
	return counts
}

func opencodeDBPath() string {
	if xdg := os.Getenv("XDG_DATA_HOME"); xdg != "" {
		return filepath.Join(xdg, "opencode", "opencode.db")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".local", "share", "opencode", "opencode.db")
}
