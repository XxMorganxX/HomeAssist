const state = {
  currentMode: "open",
  counts: { open: 0, due: 0, completed: 0 },
  currentTodos: [],
  defaultUser: "",
  refreshTimer: null,
  isEditing: false,
};

const elements = {
  userLabel: document.getElementById("userLabel"),
  listHeading: document.getElementById("listHeading"),
  statusMessage: document.getElementById("statusMessage"),
  errorBanner: document.getElementById("errorBanner"),
  emptyState: document.getElementById("emptyState"),
  todoList: document.getElementById("todoList"),
  refreshButton: document.getElementById("refreshButton"),
  addToggleButton: document.getElementById("addToggleButton"),
  addCancelButton: document.getElementById("addCancelButton"),
  addPanel: document.getElementById("addPanel"),
  createForm: document.getElementById("createForm"),
  createTitle: document.getElementById("createTitle"),
  createDueAt: document.getElementById("createDueAt"),
  createDetails: document.getElementById("createDetails"),
  openCount: document.getElementById("openCount"),
  dueCount: document.getElementById("dueCount"),
  completedCount: document.getElementById("completedCount"),
  summaryTiles: Array.from(document.querySelectorAll(".summary-tile")),
  modalBackdrop: document.getElementById("modalBackdrop"),
  editForm: document.getElementById("editForm"),
  editTodoId: document.getElementById("editTodoId"),
  editTitle: document.getElementById("editTitle"),
  editDueAt: document.getElementById("editDueAt"),
  editDetails: document.getElementById("editDetails"),
  closeModalButton: document.getElementById("closeModalButton"),
  clearDueButton: document.getElementById("clearDueButton"),
};

function setStatus(message, isError = false) {
  elements.statusMessage.textContent = message || "";
  elements.statusMessage.style.color = isError ? "#ffb4ad" : "#9ae6a4";
}

function setError(message) {
  if (!message) {
    elements.errorBanner.classList.add("hidden");
    elements.errorBanner.textContent = "";
    return;
  }

  elements.errorBanner.classList.remove("hidden");
  elements.errorBanner.textContent = message;
}

async function fetchJson(path, options = {}) {
  const url = (window.API_BASE || "") + path;
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  let payload = {};
  try {
    payload = await response.json();
  } catch (error) {
    payload = {};
  }

  if (!response.ok || payload.success === false) {
    const message = payload.error || `Request failed (${response.status})`;
    throw new Error(message);
  }

  return payload;
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function titleForMode(mode) {
  if (mode === "due") return "Due Today";
  if (mode === "completed") return "Completed Todos";
  return "Open Todos";
}

function updateCounts(counts) {
  state.counts = counts;
  elements.openCount.textContent = String(counts.open || 0);
  elements.dueCount.textContent = String(counts.due || 0);
  elements.completedCount.textContent = String(counts.completed || 0);
}

function updateActiveMode() {
  elements.listHeading.textContent = titleForMode(state.currentMode);
  for (const tile of elements.summaryTiles) {
    tile.classList.toggle("is-active", tile.dataset.mode === state.currentMode);
  }
}

function toggleAddPanel(show) {
  const visible = typeof show === "boolean" ? show : elements.addPanel.classList.contains("hidden");
  elements.addPanel.classList.toggle("hidden", !visible);
  elements.addToggleButton.classList.toggle("is-active", visible);
  if (visible) {
    elements.createTitle.focus();
  }
}

const SOURCE_LABELS = {
  manual: "Manual",
  calendar: "Calendar",
  assistant: "Assistant",
};

const SOURCE_ORDER = ["manual", "calendar", "assistant"];

function sourceLabel(key) {
  return SOURCE_LABELS[key] || key.charAt(0).toUpperCase() + key.slice(1);
}

function groupBySource(todos) {
  const groups = {};
  for (const todo of todos) {
    const key = (todo.source_type || "manual").toLowerCase();
    if (!groups[key]) groups[key] = [];
    groups[key].push(todo);
  }
  const sorted = [];
  for (const key of SOURCE_ORDER) {
    if (groups[key]) {
      sorted.push([key, groups[key]]);
      delete groups[key];
    }
  }
  for (const [key, items] of Object.entries(groups)) {
    sorted.push([key, items]);
  }
  return sorted;
}

function buildTodoCard(todo) {
  const card = document.createElement("article");
  card.className = `todo-card${todo.readonly ? " is-readonly" : ""}`;

  const isCalendar = (todo.source_type || "").toLowerCase() === "calendar";
  const detailsMarkup = (!isCalendar && todo.details)
    ? `<p class="todo-details">${escapeHtml(todo.details)}</p>`
    : "";

  const badges = [
    todo.due_display ? `<span class="badge due">${escapeHtml(todo.due_display)}</span>` : "",
    todo.readonly ? `<span class="badge readonly">Read-only</span>` : "",
  ]
    .filter(Boolean)
    .join("");

  let actionsMarkup = "";
  if (todo.can_edit) {
    actionsMarkup += `<button class="ghost-button" type="button" data-action="edit" data-todo-id="${escapeHtml(todo.id)}">Edit</button>`;
  }
  if (todo.can_toggle_complete) {
    const action = todo.completed ? "reopen" : "complete";
    const label = todo.completed ? "Reopen" : "Complete";
    actionsMarkup += `<button class="${todo.completed ? "ghost-button" : "success-button"}" type="button" data-action="${action}" data-todo-id="${escapeHtml(todo.id)}">${label}</button>`;
  }
  if (todo.can_delete) {
    actionsMarkup += `<button class="danger-button" type="button" data-action="delete" data-todo-id="${escapeHtml(todo.id)}">Delete</button>`;
  }

  card.innerHTML = `
    <div class="todo-card-header">
      <div>
        <h3 class="todo-title">${escapeHtml(todo.title || "Untitled todo")}</h3>
      </div>
    </div>
    ${detailsMarkup}
    ${badges ? `<div class="todo-meta">${badges}</div>` : ""}
    <div class="todo-actions">${actionsMarkup || `<span class="subtle">No actions available</span>`}</div>
  `;
  return card;
}

function renderTodos(todos) {
  state.currentTodos = todos;
  elements.todoList.innerHTML = "";
  elements.emptyState.classList.toggle("hidden", todos.length > 0);

  const groups = groupBySource(todos);

  for (const [sourceKey, items] of groups) {
    const section = document.createElement("details");
    section.className = "source-group";
    section.open = true;

    const prev = state.collapsedGroups && state.collapsedGroups[sourceKey];
    if (prev === true) section.open = false;

    const summary = document.createElement("summary");
    summary.className = "source-group-header";
    summary.innerHTML = `
      <span class="source-group-label">${escapeHtml(sourceLabel(sourceKey))}</span>
      <span class="source-group-count">${items.length}</span>
    `;
    section.appendChild(summary);

    const list = document.createElement("div");
    list.className = "source-group-items";
    for (const todo of items) {
      list.appendChild(buildTodoCard(todo));
    }
    section.appendChild(list);

    section.addEventListener("toggle", () => {
      if (!state.collapsedGroups) state.collapsedGroups = {};
      state.collapsedGroups[sourceKey] = !section.open;
    });

    elements.todoList.appendChild(section);
  }
}

async function loadConfig() {
  const payload = await fetchJson("/api/config");
  state.defaultUser = payload.default_user || "";
  elements.userLabel.textContent = payload.default_user
    ? `Showing todos for ${payload.default_user}`
    : "Showing current todos";
}

async function refreshSummary() {
  const payload = await fetchJson("/api/todos/summary");
  updateCounts(payload.counts || {});
}

async function refreshTodos() {
  updateActiveMode();
  const payload = await fetchJson(`/api/todos?mode=${encodeURIComponent(state.currentMode)}`);
  renderTodos(payload.todos || []);
}

async function fullRefresh({ message } = {}) {
  try {
    setError("");
    if (message) {
      setStatus(message);
    }
    await refreshSummary();
    await refreshTodos();
  } catch (error) {
    setError(error.message);
    setStatus("Unable to load todos.", true);
  }
}

function openEditModal(todo) {
  state.isEditing = true;
  elements.editTodoId.value = todo.id;
  elements.editTitle.value = todo.title || "";
  elements.editDueAt.value = todo.due_at || "";
  elements.editDetails.value = todo.details || "";
  elements.modalBackdrop.classList.remove("hidden");
}

function closeEditModal() {
  state.isEditing = false;
  elements.editForm.reset();
  elements.editTodoId.value = "";
  elements.modalBackdrop.classList.add("hidden");
}

async function createTodo(event) {
  event.preventDefault();
  const title = elements.createTitle.value.trim();
  if (!title) {
    setStatus("Title is required.", true);
    return;
  }

  const payload = {
    title,
    details: elements.createDetails.value.trim() || null,
    due_at: elements.createDueAt.value.trim() || null,
  };

  try {
    await fetchJson("/api/todos", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    elements.createForm.reset();
    toggleAddPanel(false);
    await fullRefresh({ message: `Added "${title}".` });
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function submitEdit(event) {
  event.preventDefault();
  const todoId = elements.editTodoId.value;
  if (!todoId) {
    return;
  }

  const payload = {
    title: elements.editTitle.value.trim(),
    details: elements.editDetails.value.trim(),
    due_at: elements.editDueAt.value.trim(),
  };

  if (!payload.due_at) {
    payload.clear_due_at = true;
  }

  try {
    await fetchJson(`/api/todos/${encodeURIComponent(todoId)}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    });
    closeEditModal();
    await fullRefresh({ message: "Todo updated." });
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function handleListClick(event) {
  const button = event.target.closest("button[data-action]");
  if (!button) {
    return;
  }

  const todoId = button.dataset.todoId;
  const action = button.dataset.action;
  if (!todoId || !action) {
    return;
  }

  if (action === "edit") {
    const todo = state.currentTodos.find((item) => item.id === todoId);
    if (!todo) {
      setStatus("That todo is no longer available.", true);
      return;
    }
    openEditModal(todo);
    return;
  }

  if (action === "delete" && !window.confirm("Delete this todo?")) {
    return;
  }

  try {
    if (action === "complete" || action === "reopen") {
      await fetchJson(`/api/todos/${encodeURIComponent(todoId)}/${action}`, {
        method: "POST",
        body: JSON.stringify({}),
      });
      await fullRefresh({ message: action === "complete" ? "Todo completed." : "Todo reopened." });
      return;
    }

    if (action === "delete") {
      await fetchJson(`/api/todos/${encodeURIComponent(todoId)}`, {
        method: "DELETE",
        body: JSON.stringify({}),
      });
      await fullRefresh({ message: "Todo deleted." });
    }
  } catch (error) {
    setStatus(error.message, true);
  }
}

function bindEvents() {
  elements.refreshButton.addEventListener("click", () => {
    fullRefresh({ message: "Refreshed." });
  });

  elements.addToggleButton.addEventListener("click", () => toggleAddPanel());
  elements.addCancelButton.addEventListener("click", () => toggleAddPanel(false));

  elements.createForm.addEventListener("submit", createTodo);
  elements.editForm.addEventListener("submit", submitEdit);
  elements.closeModalButton.addEventListener("click", closeEditModal);
  elements.clearDueButton.addEventListener("click", () => {
    elements.editDueAt.value = "";
  });
  elements.modalBackdrop.addEventListener("click", (event) => {
    if (event.target === elements.modalBackdrop) {
      closeEditModal();
    }
  });

  for (const tile of elements.summaryTiles) {
    tile.addEventListener("click", async () => {
      state.currentMode = tile.dataset.mode || "open";
      await fullRefresh();
    });
  }

  elements.todoList.addEventListener("click", (event) => {
    handleListClick(event);
  });
}

function startPolling() {
  if (state.refreshTimer) {
    window.clearInterval(state.refreshTimer);
  }

  state.refreshTimer = window.setInterval(() => {
    if (!state.isEditing) {
      fullRefresh();
    }
  }, 30000);
}

async function init() {
  bindEvents();
  try {
    await loadConfig();
    await fullRefresh();
    startPolling();
  } catch (error) {
    setError(error.message);
    setStatus("Unable to start the overlay.", true);
  }
}

init();
