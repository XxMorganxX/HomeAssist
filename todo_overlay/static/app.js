const state = {
  currentMode: "open",
  counts: { open: 0, due: 0, completed: 0 },
  currentTodos: [],
  availableGroups: [],
  customGroups: [],
  groupOrder: [],
  draggedGroupKey: null,
  dragOverGroupKey: null,
  defaultUser: "",
  calendarUsers: [],
  defaultCalendarUser: "",
  refreshTimer: null,
  isEditing: false,
  isRefreshing: false,
  collapsedGroups: {},
  confirmDeleteTodoId: null,
  confirmDeleteTimer: null,
};

const elements = {
  userLabel: document.getElementById("userLabel"),
  listHeading: document.getElementById("listHeading"),
  statusMessage: document.getElementById("statusMessage"),
  errorBanner: document.getElementById("errorBanner"),
  emptyState: document.getElementById("emptyState"),
  todoList: document.getElementById("todoList"),
  refreshButton: document.getElementById("refreshButton"),
  addGroupToggleButton: document.getElementById("addGroupToggleButton"),
  addGroupCancelButton: document.getElementById("addGroupCancelButton"),
  addGroupPanel: document.getElementById("addGroupPanel"),
  addToggleButton: document.getElementById("addToggleButton"),
  addCancelButton: document.getElementById("addCancelButton"),
  addPanel: document.getElementById("addPanel"),
  createGroupForm: document.getElementById("createGroupForm"),
  createGroupName: document.getElementById("createGroupName"),
  createForm: document.getElementById("createForm"),
  createTitle: document.getElementById("createTitle"),
  createDueAt: document.getElementById("createDueAt"),
  createGroup: document.getElementById("createGroup"),
  createDetails: document.getElementById("createDetails"),
  openCount: document.getElementById("openCount"),
  dueCount: document.getElementById("dueCount"),
  completedCount: document.getElementById("completedCount"),
  groupSuggestions: document.getElementById("groupSuggestions"),
  summaryTiles: Array.from(document.querySelectorAll(".summary-tile")),
  modalBackdrop: document.getElementById("modalBackdrop"),
  editForm: document.getElementById("editForm"),
  editTodoId: document.getElementById("editTodoId"),
  editTitle: document.getElementById("editTitle"),
  editDueAt: document.getElementById("editDueAt"),
  editGroup: document.getElementById("editGroup"),
  editDetails: document.getElementById("editDetails"),
  closeModalButton: document.getElementById("closeModalButton"),
  clearDueButton: document.getElementById("clearDueButton"),
  shareModalBackdrop: document.getElementById("shareModalBackdrop"),
  shareForm: document.getElementById("shareForm"),
  shareTodoId: document.getElementById("shareTodoId"),
  shareMode: document.getElementById("shareMode"),
  shareRecipients: document.getElementById("shareRecipients"),
  shareRecipientsLabel: document.getElementById("shareRecipientsLabel"),
  shareSubjectRow: document.getElementById("shareSubjectRow"),
  shareSubject: document.getElementById("shareSubject"),
  shareBodyRow: document.getElementById("shareBodyRow"),
  shareBody: document.getElementById("shareBody"),
  shareHelpText: document.getElementById("shareHelpText"),
  shareSubmitButton: document.getElementById("shareSubmitButton"),
  shareCancelButton: document.getElementById("shareCancelButton"),
  closeShareModalButton: document.getElementById("closeShareModalButton"),
  calendarModalBackdrop: document.getElementById("calendarModalBackdrop"),
  calendarForm: document.getElementById("calendarForm"),
  calendarTodoId: document.getElementById("calendarTodoId"),
  calendarOptions: document.getElementById("calendarOptions"),
  calendarSubmitButton: document.getElementById("calendarSubmitButton"),
  calendarCancelButton: document.getElementById("calendarCancelButton"),
  closeCalendarModalButton: document.getElementById("closeCalendarModalButton"),
  moveModalBackdrop: document.getElementById("moveModalBackdrop"),
  moveForm: document.getElementById("moveForm"),
  moveTodoId: document.getElementById("moveTodoId"),
  moveGroupSelect: document.getElementById("moveGroupSelect"),
  moveSubmitButton: document.getElementById("moveSubmitButton"),
  moveCancelButton: document.getElementById("moveCancelButton"),
  closeMoveModalButton: document.getElementById("closeMoveModalButton"),
  advanceModalBackdrop: document.getElementById("advanceModalBackdrop"),
  advanceForm: document.getElementById("advanceForm"),
  advanceTodoId: document.getElementById("advanceTodoId"),
  advanceTitle: document.getElementById("advanceTitle"),
  advanceDueAt: document.getElementById("advanceDueAt"),
  advanceDetails: document.getElementById("advanceDetails"),
  advancePreviousTitle: document.getElementById("advancePreviousTitle"),
  advanceSubmitButton: document.getElementById("advanceSubmitButton"),
  advanceCancelButton: document.getElementById("advanceCancelButton"),
  closeAdvanceModalButton: document.getElementById("closeAdvanceModalButton"),
  chainModalBackdrop: document.getElementById("chainModalBackdrop"),
  chainModalSubtitle: document.getElementById("chainModalSubtitle"),
  chainTimeline: document.getElementById("chainTimeline"),
  closeChainModalButton: document.getElementById("closeChainModalButton"),
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

function setButtonBusy(button, busy, busyLabel = "Working...") {
  if (!button) return;
  if (busy) {
    if (!button.dataset.originalLabel) {
      button.dataset.originalLabel = button.textContent;
    }
    button.disabled = true;
    button.textContent = busyLabel;
    button.classList.add("is-busy");
    return;
  }

  if (button.dataset.originalLabel) {
    button.textContent = button.dataset.originalLabel;
    delete button.dataset.originalLabel;
  }
  button.disabled = false;
  button.classList.remove("is-busy");
}

function findDeleteButton(todoId) {
  return Array.from(elements.todoList.querySelectorAll('button[data-action="delete"]'))
    .find((button) => button.dataset.todoId === todoId) || null;
}

function clearDeleteConfirmation() {
  const previousTodoId = state.confirmDeleteTodoId;
  if (state.confirmDeleteTimer) {
    window.clearTimeout(state.confirmDeleteTimer);
    state.confirmDeleteTimer = null;
  }
  state.confirmDeleteTodoId = null;
  if (!previousTodoId) {
    return;
  }

  const button = findDeleteButton(previousTodoId);
  if (button) {
    button.textContent = "Delete";
  }
}

function buildTodoMessage(todo) {
  const lines = [todo.title || "Untitled todo"];
  if (todo.details) {
    lines.push("");
    lines.push(todo.details);
  }
  if (todo.due_display) {
    lines.push("");
    lines.push(`Due: ${todo.due_display}`);
  }
  return lines.join("\n");
}

function normalizeShareText(text) {
  return String(text || "")
    .replaceAll("+", " ")
    .replace(/\s+/g, " ")
    .trim();
}

function normalizeGroupName(value) {
  return String(value || "")
    .replace(/\s+/g, " ")
    .trim();
}

function prettifyTodoTitle(text) {
  const normalized = normalizeShareText(text);
  if (!normalized) return "Untitled Todo";

  return normalized
    .split(" ")
    .map((word) => {
      if (!word) return word;
      if (/^[A-Z0-9&/-]+$/.test(word)) return word;
      return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
    })
    .join(" ");
}

function prettifyCalendarUser(value) {
  const normalized = String(value || "").trim();
  if (!normalized) return "Calendar";

  return normalized
    .split("_")
    .map((part) => {
      if (!part) return part;
      if (part.toLowerCase() === "ai") return "AI";
      return part.charAt(0).toUpperCase() + part.slice(1);
    })
    .join(" ");
}

function formatInviteDate(todo) {
  if (!todo.due_at) {
    return "";
  }

  const dueDate = new Date(todo.due_at);
  if (Number.isNaN(dueDate.getTime())) {
    return normalizeShareText(todo.due_display || "");
  }

  return new Intl.DateTimeFormat(undefined, {
    weekday: "long",
    month: "long",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
    timeZoneName: "short",
  }).format(dueDate);
}

function buildInviteEmailDraft(todo) {
  const subject = prettifyTodoTitle(todo.title);
  const formattedDate = formatInviteDate(todo);
  const message = formattedDate
    ? `I'd like to invite you to the ${subject} on ${formattedDate}.`
    : `I'd like to invite you to the ${subject}.`;

  return { subject, message };
}

function parseRecipients(value) {
  return String(value || "")
    .split(/[\n,;]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function encodeMailtoComponent(value) {
  return encodeURIComponent(String(value || ""));
}

function buildMailtoUrl(recipients, subject, body) {
  const queryParts = [];
  if (subject) {
    queryParts.push(`subject=${encodeMailtoComponent(subject)}`);
  }
  if (body) {
    queryParts.push(`body=${encodeMailtoComponent(body)}`);
  }
  const query = queryParts.length ? `?${queryParts.join("&")}` : "";
  return `mailto:${recipients.map((recipient) => encodeMailtoComponent(recipient)).join(",")}${query}`;
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

function cloneTodos(todos) {
  return todos.map((todo) => ({ ...todo }));
}

function optimisticCounts(todo, action) {
  const counts = { ...state.counts };

  if (action === "complete") {
    counts.open = Math.max(0, (counts.open || 0) - 1);
    counts.completed = (counts.completed || 0) + 1;
    if (state.currentMode === "due") {
      counts.due = Math.max(0, (counts.due || 0) - 1);
    }
  } else if (action === "reopen") {
    counts.open = (counts.open || 0) + 1;
    counts.completed = Math.max(0, (counts.completed || 0) - 1);
  } else if (action === "delete") {
    if (todo.completed) {
      counts.completed = Math.max(0, (counts.completed || 0) - 1);
    } else {
      counts.open = Math.max(0, (counts.open || 0) - 1);
      if (state.currentMode === "due") {
        counts.due = Math.max(0, (counts.due || 0) - 1);
      }
    }
  }

  return counts;
}

function optimisticTodos(todoId, action) {
  let todos = cloneTodos(state.currentTodos);

  if (action === "delete") {
    return todos.filter((todo) => todo.id !== todoId);
  }

  todos = todos.map((todo) => {
    if (todo.id !== todoId) return todo;
    return {
      ...todo,
      completed: action === "complete",
    };
  });

  if ((action === "complete" && state.currentMode !== "completed")
    || (action === "reopen" && state.currentMode === "completed")) {
    todos = todos.filter((todo) => todo.id !== todoId);
  }

  return todos;
}

function updateActiveMode() {
  elements.listHeading.textContent = titleForMode(state.currentMode);
  for (const tile of elements.summaryTiles) {
    tile.classList.toggle("is-active", tile.dataset.mode === state.currentMode);
  }
}

function toggleAddPanel(show) {
  const visible = typeof show === "boolean" ? show : elements.addPanel.classList.contains("hidden");
  if (visible) {
    elements.addGroupPanel.classList.add("hidden");
    elements.addGroupToggleButton.classList.remove("is-active");
  }
  elements.addPanel.classList.toggle("hidden", !visible);
  elements.addToggleButton.classList.toggle("is-active", visible);
  if (visible) {
    elements.createTitle.focus();
  }
}

function toggleAddGroupPanel(show) {
  const visible = typeof show === "boolean" ? show : elements.addGroupPanel.classList.contains("hidden");
  if (visible) {
    elements.addPanel.classList.add("hidden");
    elements.addToggleButton.classList.remove("is-active");
  }
  elements.addGroupPanel.classList.toggle("hidden", !visible);
  elements.addGroupToggleButton.classList.toggle("is-active", visible);
  if (visible) {
    elements.createGroupName.focus();
  }
}

const UNGROUPED_GROUP_KEY = "__ungrouped__";
const UNGROUPED_GROUP_LABEL = "Ungrouped";
const CALENDAR_GROUP_KEY = "__calendar__";
const CALENDAR_GROUP_LABEL = "Calendar";
const GROUP_ORDER_STORAGE_PREFIX = "homeassist.todo.groupOrder";
const CUSTOM_GROUPS_STORAGE_PREFIX = "homeassist.todo.customGroups";

function uniqueGroupKeys(keys) {
  const seen = new Set();
  const unique = [];
  for (const key of keys || []) {
    const normalized = String(key || "").trim();
    if (!normalized || seen.has(normalized)) {
      continue;
    }
    seen.add(normalized);
    unique.push(normalized);
  }
  return unique;
}

function groupOrderStorageKey() {
  const userKey = String(state.defaultUser || "default")
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "_");
  return `${GROUP_ORDER_STORAGE_PREFIX}:${userKey}`;
}

function customGroupsStorageKey() {
  const userKey = String(state.defaultUser || "default")
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "_");
  return `${CUSTOM_GROUPS_STORAGE_PREFIX}:${userKey}`;
}

function loadGroupOrder() {
  state.groupOrder = [];
  try {
    const raw = window.localStorage.getItem(groupOrderStorageKey());
    if (!raw) {
      return;
    }
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return;
    }
    state.groupOrder = uniqueGroupKeys(parsed);
  } catch (_error) {
    state.groupOrder = [];
  }
}

function saveGroupOrder() {
  try {
    window.localStorage.setItem(groupOrderStorageKey(), JSON.stringify(uniqueGroupKeys(state.groupOrder)));
  } catch (_error) {
    // Ignore storage errors in restricted browser contexts.
  }
}

function loadCustomGroups() {
  state.customGroups = [];
  try {
    const raw = window.localStorage.getItem(customGroupsStorageKey());
    if (!raw) {
      return;
    }
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return;
    }
    state.customGroups = sortedUniqueGroupNames(parsed);
  } catch (_error) {
    state.customGroups = [];
  }
}

function saveCustomGroups() {
  try {
    window.localStorage.setItem(customGroupsStorageKey(), JSON.stringify(sortedUniqueGroupNames(state.customGroups)));
  } catch (_error) {
    // Ignore storage errors in restricted browser contexts.
  }
}

function groupKeyFromName(name) {
  const normalized = normalizeGroupName(name);
  return normalized ? normalized.toLowerCase() : UNGROUPED_GROUP_KEY;
}

function sortedUniqueGroupNames(values) {
  const seen = new Map();
  for (const value of values || []) {
    const normalized = normalizeGroupName(value);
    if (!normalized) continue;
    const key = normalized.toLowerCase();
    if (!seen.has(key)) {
      seen.set(key, normalized);
    }
  }
  return Array.from(seen.values()).sort((left, right) => (
    left.localeCompare(right, undefined, { sensitivity: "base" })
  ));
}

function upsertAvailableGroups(values) {
  state.availableGroups = sortedUniqueGroupNames([...(values || []), ...(state.customGroups || [])]);
  renderGroupSuggestions();
}

function renderGroupSuggestions() {
  if (!elements.groupSuggestions) {
    return;
  }
  elements.groupSuggestions.innerHTML = "";
  for (const groupName of state.availableGroups) {
    const option = document.createElement("option");
    option.value = groupName;
    elements.groupSuggestions.appendChild(option);
  }
}

function mergeGroupsFromTodos(todos) {
  const merged = [...state.availableGroups];
  for (const todo of todos) {
    if ((todo.source_type || "").toLowerCase() === "calendar") {
      continue;
    }
    const groupName = normalizeGroupName(todo.group);
    if (groupName) {
      merged.push(groupName);
    }
  }
  upsertAvailableGroups(merged);
}

function addCustomGroup(groupName) {
  const normalized = normalizeGroupName(groupName);
  if (!normalized) {
    return { success: false, reason: "Group name cannot be empty." };
  }
  if (normalized.toLowerCase() === CALENDAR_GROUP_LABEL.toLowerCase()) {
    return { success: false, reason: "Calendar is reserved for synced calendar events." };
  }

  const exists = state.customGroups.some((group) => group.toLowerCase() === normalized.toLowerCase());
  if (!exists) {
    state.customGroups = [...state.customGroups, normalized];
    saveCustomGroups();
  }

  upsertAvailableGroups(state.availableGroups);

  const groupKey = groupKeyFromName(normalized);
  if (!state.groupOrder.includes(groupKey)) {
    const calendarIndex = state.groupOrder.indexOf(CALENDAR_GROUP_KEY);
    if (calendarIndex >= 0) {
      state.groupOrder.splice(calendarIndex, 0, groupKey);
    } else {
      state.groupOrder.push(groupKey);
    }
    saveGroupOrder();
  }

  return { success: true, reason: exists ? "exists" : "created", name: normalized };
}

function groupForTodo(todo) {
  if ((todo.source_type || "").toLowerCase() === "calendar") {
    return {
      key: CALENDAR_GROUP_KEY,
      label: CALENDAR_GROUP_LABEL,
    };
  }
  const groupName = normalizeGroupName(todo.group);
  return {
    key: groupKeyFromName(groupName),
    label: groupName || UNGROUPED_GROUP_LABEL,
  };
}

function defaultGroupComparator(left, right, orderedGroups) {
  if (left.key === CALENDAR_GROUP_KEY && right.key !== CALENDAR_GROUP_KEY) return 1;
  if (right.key === CALENDAR_GROUP_KEY && left.key !== CALENDAR_GROUP_KEY) return -1;

  if (left.key === UNGROUPED_GROUP_KEY && right.key !== UNGROUPED_GROUP_KEY) return 1;
  if (right.key === UNGROUPED_GROUP_KEY && left.key !== UNGROUPED_GROUP_KEY) return -1;

  const leftOrder = orderedGroups.get(left.key);
  const rightOrder = orderedGroups.get(right.key);
  if (leftOrder !== undefined && rightOrder !== undefined) return leftOrder - rightOrder;
  if (leftOrder !== undefined) return -1;
  if (rightOrder !== undefined) return 1;
  return left.label.localeCompare(right.label, undefined, { sensitivity: "base" });
}

function buildGroupDisplayOrder(groups, orderedGroups) {
  const defaultOrder = groups
    .slice()
    .sort((left, right) => defaultGroupComparator(left, right, orderedGroups))
    .map((group) => group.key);

  if (!state.groupOrder.length) {
    state.groupOrder = [...defaultOrder];
    saveGroupOrder();
  }

  let changed = false;
  const known = new Set(state.groupOrder);
  for (const key of defaultOrder) {
    if (!known.has(key)) {
      state.groupOrder.push(key);
      known.add(key);
      changed = true;
    }
  }
  if (changed) {
    saveGroupOrder();
  }

  const present = new Set(defaultOrder);
  const orderedPresent = state.groupOrder.filter((key) => present.has(key));
  for (const key of defaultOrder) {
    if (!orderedPresent.includes(key)) {
      orderedPresent.push(key);
    }
  }
  return orderedPresent;
}

function moveGroupInOrder(draggedKey, targetKey) {
  const dragged = String(draggedKey || "").trim();
  const target = String(targetKey || "").trim();
  if (!dragged || !target || dragged === target) {
    return false;
  }

  const order = uniqueGroupKeys(state.groupOrder);
  if (!order.includes(dragged)) {
    order.push(dragged);
  }
  if (!order.includes(target)) {
    order.push(target);
  }

  const fromIndex = order.indexOf(dragged);
  const toIndex = order.indexOf(target);
  if (fromIndex < 0 || toIndex < 0) {
    return false;
  }

  order.splice(fromIndex, 1);
  const insertIndex = fromIndex < toIndex ? toIndex - 1 : toIndex;
  order.splice(insertIndex, 0, dragged);

  state.groupOrder = order;
  saveGroupOrder();
  return true;
}

function clearDragIndicators() {
  for (const node of elements.todoList.querySelectorAll(".source-group.is-drop-target, .source-group.is-dragging")) {
    node.classList.remove("is-drop-target");
    node.classList.remove("is-dragging");
  }
}

function setDragOverGroup(groupKey) {
  state.dragOverGroupKey = groupKey;
  clearDragIndicators();
  if (!groupKey) {
    return;
  }
  for (const section of elements.todoList.querySelectorAll(".source-group")) {
    if (section.dataset.groupKey === groupKey) {
      section.classList.add("is-drop-target");
      break;
    }
  }
}

function resetGroupDragState() {
  state.draggedGroupKey = null;
  state.dragOverGroupKey = null;
  clearDragIndicators();
}

function groupByCustomGroup(todos, { includeEmptyGroups = false } = {}) {
  const grouped = new Map();
  const orderedGroups = new Map();
  state.availableGroups.forEach((name, index) => {
    orderedGroups.set(groupKeyFromName(name), index);
  });

  for (const todo of todos) {
    const group = groupForTodo(todo);
    const key = group.key;
    if (!grouped.has(key)) {
      grouped.set(key, {
        key,
        label: group.label,
        items: [],
      });
    }
    grouped.get(key).items.push(todo);
  }

  if (includeEmptyGroups) {
    for (const groupName of state.availableGroups) {
      const key = groupKeyFromName(groupName);
      if (!grouped.has(key)) {
        grouped.set(key, {
          key,
          label: groupName,
          items: [],
        });
      }
    }
  }

  const groupedList = Array.from(grouped.values());
  const displayOrder = buildGroupDisplayOrder(groupedList, orderedGroups);
  const displayIndex = new Map(displayOrder.map((key, index) => [key, index]));

  return groupedList.sort((left, right) => (
    (displayIndex.get(left.key) ?? Number.MAX_SAFE_INTEGER)
    - (displayIndex.get(right.key) ?? Number.MAX_SAFE_INTEGER)
  ));
}

function syncCollapsedGroups(groups) {
  const nextCollapsed = {};
  for (const group of groups) {
    if (state.collapsedGroups[group.key] === true) {
      nextCollapsed[group.key] = true;
    }
  }
  state.collapsedGroups = nextCollapsed;
}

function buildTodoCard(todo) {
  const card = document.createElement("article");
  card.className = `todo-card${todo.readonly ? " is-readonly" : ""}`;

  const isCalendar = (todo.source_type || "").toLowerCase() === "calendar";
  const detailsMarkup = (!isCalendar && todo.details)
    ? `<p class="todo-details">${escapeHtml(todo.details)}</p>`
    : "";

  const chainBadge = todo.is_chain
    ? `<span class="badge chain" data-action="chain" data-todo-id="${escapeHtml(todo.id)}" title="View chain history">Step ${todo.chain_position || 1}</span>`
    : "";

  const badges = [
    chainBadge,
    todo.due_display ? `<span class="badge due">${escapeHtml(todo.due_display)}</span>` : "",
    todo.readonly ? `<span class="badge readonly">Read-only</span>` : "",
    todo.calendar_linked ? `<span class="badge">On Calendar</span>` : "",
  ]
    .filter(Boolean)
    .join("");

  let actionsMarkup = "";
  if (todo.can_edit) {
    actionsMarkup += `<button class="ghost-button" type="button" data-action="edit" data-todo-id="${escapeHtml(todo.id)}">Edit</button>`;
    actionsMarkup += `<button class="ghost-button" type="button" data-action="move" data-todo-id="${escapeHtml(todo.id)}">Move</button>`;
  }
  if (todo.can_toggle_complete) {
    const action = todo.completed ? "reopen" : "complete";
    const label = todo.completed ? "Reopen" : "Complete";
    actionsMarkup += `<button class="${todo.completed ? "ghost-button" : "success-button"}" type="button" data-action="${action}" data-todo-id="${escapeHtml(todo.id)}">${label}</button>`;
  }
  if (todo.can_send_email) {
    actionsMarkup += `<button class="ghost-button" type="button" data-action="email" data-todo-id="${escapeHtml(todo.id)}">Send Email</button>`;
  }
  if (todo.can_add_to_calendar) {
    actionsMarkup += `<button class="ghost-button" type="button" data-action="calendar" data-todo-id="${escapeHtml(todo.id)}">Add to Calendar</button>`;
  }
  if (todo.can_create_invite) {
    actionsMarkup += `<button class="ghost-button" type="button" data-action="invite" data-todo-id="${escapeHtml(todo.id)}">Create Invite</button>`;
  }
  if (todo.can_advance) {
    actionsMarkup += `<button class="advance-button" type="button" data-action="advance" data-todo-id="${escapeHtml(todo.id)}">Next Step</button>`;
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
  resetGroupDragState();
  clearDeleteConfirmation();
  state.currentTodos = todos;
  elements.todoList.innerHTML = "";
  mergeGroupsFromTodos(todos);

  const groups = groupByCustomGroup(todos, { includeEmptyGroups: state.currentMode === "open" });
  elements.emptyState.classList.toggle("hidden", groups.length > 0);
  syncCollapsedGroups(groups);

  for (const group of groups) {
    const section = document.createElement("details");
    section.className = "source-group";
    section.dataset.groupKey = group.key;
    section.open = state.collapsedGroups[group.key] !== true;

    const summary = document.createElement("summary");
    summary.className = "source-group-header";
    summary.draggable = true;
    summary.title = "Drag to reorder folders";
    summary.innerHTML = `
      <span class="source-group-label">${escapeHtml(group.label)}</span>
      <span class="source-group-count">${group.items.length}</span>
    `;

    summary.addEventListener("dragstart", (event) => {
      state.draggedGroupKey = group.key;
      section.classList.add("is-dragging");
      if (event.dataTransfer) {
        event.dataTransfer.effectAllowed = "move";
        event.dataTransfer.setData("text/plain", group.key);
      }
    });

    summary.addEventListener("dragover", (event) => {
      const dragged = state.draggedGroupKey;
      if (!dragged || dragged === group.key) {
        return;
      }
      event.preventDefault();
      if (event.dataTransfer) {
        event.dataTransfer.dropEffect = "move";
      }
      setDragOverGroup(group.key);
    });

    summary.addEventListener("drop", (event) => {
      event.preventDefault();
      const dragged = state.draggedGroupKey
        || (event.dataTransfer ? event.dataTransfer.getData("text/plain") : "");
      const moved = moveGroupInOrder(dragged, group.key);
      resetGroupDragState();
      if (moved) {
        renderTodos(state.currentTodos);
      }
    });

    summary.addEventListener("dragend", () => {
      resetGroupDragState();
    });

    section.appendChild(summary);

    const list = document.createElement("div");
    list.className = "source-group-items";
    if (group.items.length === 0) {
      const emptyGroup = document.createElement("p");
      emptyGroup.className = "source-group-empty";
      emptyGroup.textContent = "No todos in this group yet.";
      list.appendChild(emptyGroup);
    } else {
      for (const todo of group.items) {
        list.appendChild(buildTodoCard(todo));
      }
    }
    section.appendChild(list);

    section.addEventListener("toggle", () => {
      state.collapsedGroups[group.key] = !section.open;
    });

    elements.todoList.appendChild(section);
  }
}

async function loadConfig() {
  const payload = await fetchJson("/api/config");
  state.defaultUser = payload.default_user || "";
  loadCustomGroups();
  loadGroupOrder();
  state.calendarUsers = Array.isArray(payload.calendar_users) ? payload.calendar_users : [];
  state.defaultCalendarUser = payload.default_calendar_user || "";
  upsertAvailableGroups([]);
  elements.userLabel.textContent = payload.default_user
    ? `Showing todos for ${payload.default_user}`
    : "Showing current todos";
}

async function refreshSummary() {
  const payload = await fetchJson("/api/todos/summary");
  updateCounts(payload.counts || {});
}

async function refreshGroups() {
  const payload = await fetchJson("/api/todos/groups");
  upsertAvailableGroups(payload.groups || []);
}

async function refreshTodos() {
  updateActiveMode();
  const payload = await fetchJson(`/api/todos?mode=${encodeURIComponent(state.currentMode)}`);
  renderTodos(payload.todos || []);
}

async function fullRefresh({ message } = {}) {
  if (state.isRefreshing) {
    return;
  }

  try {
    state.isRefreshing = true;
    setError("");
    if (message) {
      setStatus(message);
    }
    setButtonBusy(elements.refreshButton, true, "...");
    await Promise.all([refreshSummary(), refreshGroups()]);
    await refreshTodos();
  } catch (error) {
    setError(error.message);
    setStatus("Unable to load todos.", true);
  } finally {
    state.isRefreshing = false;
    setButtonBusy(elements.refreshButton, false);
  }
}

function openEditModal(todo) {
  state.isEditing = true;
  elements.editTodoId.value = todo.id;
  elements.editTitle.value = todo.title || "";
  elements.editDueAt.value = todo.due_at || "";
  elements.editGroup.value = todo.group || "";
  elements.editDetails.value = todo.details || "";
  elements.modalBackdrop.classList.remove("hidden");
}

function closeEditModal() {
  state.isEditing = false;
  elements.editForm.reset();
  elements.editTodoId.value = "";
  elements.modalBackdrop.classList.add("hidden");
}

function openShareModal(todo, mode) {
  state.isEditing = true;
  elements.shareForm.reset();
  elements.shareTodoId.value = todo.id;
  elements.shareMode.value = mode;
  elements.shareRecipients.value = "";

  if (mode === "email") {
    const draft = buildInviteEmailDraft(todo);
    elements.shareRecipientsLabel.textContent = "To";
    elements.shareHelpText.textContent = "Opens your default mail app with a draft based on this todo.";
    elements.shareSubjectRow.classList.remove("hidden");
    elements.shareBodyRow.classList.remove("hidden");
    elements.shareSubject.value = draft.subject;
    elements.shareBody.value = draft.message;
    elements.shareSubmitButton.textContent = "Open Email Draft";
  } else {
    elements.shareRecipientsLabel.textContent = "Invite Emails";
    elements.shareHelpText.textContent = "Creates a Google Calendar event from this todo and sends invites to these attendees.";
    elements.shareSubjectRow.classList.add("hidden");
    elements.shareBodyRow.classList.add("hidden");
    elements.shareSubject.value = "";
    elements.shareBody.value = "";
    elements.shareSubmitButton.textContent = "Send Invite";
  }

  elements.shareModalBackdrop.classList.remove("hidden");
  elements.shareRecipients.focus();
}

function closeShareModal() {
  state.isEditing = false;
  elements.shareForm.reset();
  elements.shareTodoId.value = "";
  elements.shareMode.value = "";
  elements.shareSubjectRow.classList.remove("hidden");
  elements.shareBodyRow.classList.remove("hidden");
  elements.shareModalBackdrop.classList.add("hidden");
}

function openCalendarModal(todo) {
  state.isEditing = true;
  elements.calendarForm.reset();
  elements.calendarTodoId.value = todo.id;
  elements.calendarOptions.innerHTML = "";

  const missingUsers = Array.isArray(todo.missing_calendar_users) && todo.missing_calendar_users.length
    ? todo.missing_calendar_users
    : state.calendarUsers;
  const linkedUsers = Array.isArray(todo.linked_calendar_users) ? todo.linked_calendar_users : [];
  const defaultSelections = new Set(
    missingUsers.includes(state.defaultCalendarUser)
      ? [state.defaultCalendarUser]
      : missingUsers
  );

  for (const calendarUser of missingUsers) {
    const row = document.createElement("label");
    row.className = "calendar-option";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.name = "calendar_user";
    checkbox.value = calendarUser;
    checkbox.checked = defaultSelections.has(calendarUser);

    const text = document.createElement("div");
    text.className = "calendar-option-text";

    const title = document.createElement("span");
    title.className = "calendar-option-title";
    title.textContent = prettifyCalendarUser(calendarUser);

    const hint = document.createElement("span");
    hint.className = "calendar-option-hint";
    hint.textContent = linkedUsers.includes(calendarUser)
      ? "Already linked"
      : `Adds this todo to ${calendarUser}`;

    text.appendChild(title);
    text.appendChild(hint);
    row.appendChild(checkbox);
    row.appendChild(text);
    elements.calendarOptions.appendChild(row);
  }

  elements.calendarModalBackdrop.classList.remove("hidden");
}

function closeCalendarModal() {
  state.isEditing = false;
  elements.calendarForm.reset();
  elements.calendarTodoId.value = "";
  elements.calendarOptions.innerHTML = "";
  elements.calendarModalBackdrop.classList.add("hidden");
}

function openMoveModal(todo) {
  state.isEditing = true;
  elements.moveForm.reset();
  elements.moveTodoId.value = todo.id;
  elements.moveGroupSelect.innerHTML = "";

  const currentGroup = normalizeGroupName(todo.group);
  const options = sortedUniqueGroupNames([...(state.availableGroups || []), ...(currentGroup ? [currentGroup] : [])]);
  const groupsWithUngrouped = ["", ...options];

  for (const groupName of groupsWithUngrouped) {
    const option = document.createElement("option");
    option.value = groupName;
    option.textContent = groupName || "Ungrouped";
    option.selected = groupName === (currentGroup || "");
    elements.moveGroupSelect.appendChild(option);
  }

  elements.moveModalBackdrop.classList.remove("hidden");
  elements.moveGroupSelect.focus();
}

function closeMoveModal() {
  state.isEditing = false;
  elements.moveForm.reset();
  elements.moveTodoId.value = "";
  elements.moveGroupSelect.innerHTML = "";
  elements.moveModalBackdrop.classList.add("hidden");
}

async function submitMove(event) {
  event.preventDefault();
  const submitButton = event.submitter || elements.moveSubmitButton;
  const todoId = elements.moveTodoId.value;
  if (!todoId) {
    return;
  }

  const selectedGroup = normalizeGroupName(elements.moveGroupSelect.value);
  const payload = selectedGroup
    ? { group: selectedGroup }
    : { clear_group: true };

  try {
    setButtonBusy(submitButton, true, "Moving...");
    await fetchJson(`/api/todos/${encodeURIComponent(todoId)}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    });
    closeMoveModal();
    await fullRefresh({ message: selectedGroup ? `Moved to "${selectedGroup}".` : "Moved to Ungrouped." });
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    setButtonBusy(submitButton, false);
  }
}

function openAdvanceModal(todo) {
  state.isEditing = true;
  elements.advanceForm.reset();
  elements.advanceTodoId.value = todo.id;
  elements.advancePreviousTitle.textContent = todo.title || "Untitled todo";
  elements.advanceTitle.value = "";
  elements.advanceDueAt.value = "";
  elements.advanceDetails.value = "";
  elements.advanceModalBackdrop.classList.remove("hidden");
  elements.advanceTitle.focus();
}

function closeAdvanceModal() {
  state.isEditing = false;
  elements.advanceForm.reset();
  elements.advanceTodoId.value = "";
  elements.advanceModalBackdrop.classList.add("hidden");
}

async function submitAdvance(event) {
  event.preventDefault();
  const submitButton = event.submitter || elements.advanceSubmitButton;
  const todoId = elements.advanceTodoId.value;
  if (!todoId) return;

  const title = elements.advanceTitle.value.trim();
  if (!title) {
    setStatus("Title is required for the next step.", true);
    return;
  }

  const payload = {
    title,
    details: elements.advanceDetails.value.trim() || null,
    due_at: elements.advanceDueAt.value.trim() || null,
  };

  try {
    setButtonBusy(submitButton, true, "Advancing...");
    await fetchJson(`/api/todos/${encodeURIComponent(todoId)}/advance`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    closeAdvanceModal();
    await fullRefresh({ message: "Advanced to next step." });
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    setButtonBusy(submitButton, false);
  }
}

async function openChainModal(todoId) {
  state.isEditing = true;
  elements.chainTimeline.innerHTML = '<p class="subtle" style="padding:12px;text-align:center;">Loading chain…</p>';
  elements.chainModalBackdrop.classList.remove("hidden");

  try {
    const payload = await fetchJson(`/api/todos/${encodeURIComponent(todoId)}/chain`);
    const chain = payload.chain || [];
    if (chain.length === 0) {
      elements.chainTimeline.innerHTML = '<p class="subtle" style="padding:12px;text-align:center;">No chain history found.</p>';
      return;
    }

    elements.chainModalSubtitle.textContent = `${chain.length} step${chain.length !== 1 ? "s" : ""} in this chain.`;
    elements.chainTimeline.innerHTML = "";

    for (const step of chain) {
      const metadata = step.source_metadata || {};
      const position = metadata.chain_position || step.chain_position || 1;
      const isCompleted = Boolean(step.completed);
      const isCurrent = !isCompleted && !metadata.advanced_to;
      const statusClass = isCurrent ? "is-current" : isCompleted ? "is-completed" : "";

      const stepEl = document.createElement("div");
      stepEl.className = `chain-step ${statusClass}`;

      const metaParts = [];
      if (step.completed_at) {
        const completedDate = new Date(step.completed_at);
        if (!Number.isNaN(completedDate.getTime())) {
          metaParts.push(`Completed ${completedDate.toLocaleDateString()}`);
        }
      } else if (step.due_display) {
        metaParts.push(`Due: ${escapeHtml(step.due_display)}`);
      }
      if (step.created_at) {
        const createdDate = new Date(step.created_at);
        if (!Number.isNaN(createdDate.getTime())) {
          metaParts.push(`Created ${createdDate.toLocaleDateString()}`);
        }
      }

      const statusLabel = isCurrent ? "Current" : isCompleted ? "Done" : "Pending";

      stepEl.innerHTML = `
        <div class="chain-step-indicator">
          <div class="chain-step-dot"></div>
          <div class="chain-step-line"></div>
        </div>
        <div class="chain-step-content">
          <span class="chain-step-position">Step ${position} · ${statusLabel}</span>
          <div class="chain-step-title">${escapeHtml(step.title || "Untitled")}</div>
          ${metaParts.length ? `<div class="chain-step-meta">${metaParts.map((p) => `<span>${p}</span>`).join("")}</div>` : ""}
        </div>
      `;
      elements.chainTimeline.appendChild(stepEl);
    }
  } catch (error) {
    elements.chainTimeline.innerHTML = `<p class="subtle" style="padding:12px;text-align:center;color:#ffb4ad;">${escapeHtml(error.message)}</p>`;
  }
}

function closeChainModal() {
  state.isEditing = false;
  elements.chainTimeline.innerHTML = "";
  elements.chainModalBackdrop.classList.add("hidden");
}

async function createTodo(event) {
  event.preventDefault();
  const submitButton = event.submitter || elements.createForm.querySelector('button[type="submit"]');
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
  const groupName = normalizeGroupName(elements.createGroup.value);
  if (groupName) {
    payload.group = groupName;
  }

  try {
    setButtonBusy(submitButton, true, "Adding...");
    await fetchJson("/api/todos", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    elements.createForm.reset();
    toggleAddPanel(false);
    await fullRefresh({ message: `Added "${title}".` });
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    setButtonBusy(submitButton, false);
  }
}

async function createGroup(event) {
  event.preventDefault();
  const submitButton = event.submitter || elements.createGroupForm.querySelector('button[type="submit"]');
  const proposedName = normalizeGroupName(elements.createGroupName.value);

  if (!proposedName) {
    setStatus("Group name is required.", true);
    return;
  }

  try {
    setButtonBusy(submitButton, true, "Creating...");
    const result = addCustomGroup(proposedName);
    if (!result.success) {
      setStatus(result.reason || "Unable to create group.", true);
      return;
    }

    elements.createGroupForm.reset();
    toggleAddGroupPanel(false);
    renderTodos(state.currentTodos);
    if (result.reason === "exists") {
      setStatus(`"${result.name}" already exists.`);
    } else {
      setStatus(`Created group "${result.name}".`);
    }
  } finally {
    setButtonBusy(submitButton, false);
  }
}

async function submitEdit(event) {
  event.preventDefault();
  const submitButton = event.submitter || elements.editForm.querySelector('button[type="submit"]');
  const todoId = elements.editTodoId.value;
  if (!todoId) {
    return;
  }

  const payload = {
    title: elements.editTitle.value.trim(),
    details: elements.editDetails.value.trim(),
    due_at: elements.editDueAt.value.trim(),
    group: normalizeGroupName(elements.editGroup.value),
  };

  if (!payload.due_at) {
    payload.clear_due_at = true;
  }
  if (!payload.group) {
    payload.clear_group = true;
  }

  try {
    setButtonBusy(submitButton, true, "Saving...");
    await fetchJson(`/api/todos/${encodeURIComponent(todoId)}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    });
    closeEditModal();
    await fullRefresh({ message: "Todo updated." });
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    setButtonBusy(submitButton, false);
  }
}

async function handleListClick(event) {
  const button = event.target.closest("[data-action]");
  if (!button) {
    return;
  }

  const todoId = button.dataset.todoId;
  const action = button.dataset.action;
  if (!todoId || !action) {
    return;
  }

  if (action === "edit") {
    clearDeleteConfirmation();
    const todo = state.currentTodos.find((item) => item.id === todoId);
    if (!todo) {
      setStatus("That todo is no longer available.", true);
      return;
    }
    openEditModal(todo);
    return;
  }

  if (action === "move") {
    clearDeleteConfirmation();
    const todo = state.currentTodos.find((item) => item.id === todoId);
    if (!todo) {
      setStatus("That todo is no longer available.", true);
      return;
    }
    openMoveModal(todo);
    return;
  }

  if (action === "advance") {
    clearDeleteConfirmation();
    const todo = state.currentTodos.find((item) => item.id === todoId);
    if (!todo) {
      setStatus("That todo is no longer available.", true);
      return;
    }
    openAdvanceModal(todo);
    return;
  }

  if (action === "chain") {
    clearDeleteConfirmation();
    openChainModal(todoId);
    return;
  }

  if (action === "email" || action === "invite") {
    clearDeleteConfirmation();
    const todo = state.currentTodos.find((item) => item.id === todoId);
    if (!todo) {
      setStatus("That todo is no longer available.", true);
      return;
    }
    openShareModal(todo, action);
    return;
  }

  const todo = state.currentTodos.find((item) => item.id === todoId);
  if (!todo) {
    clearDeleteConfirmation();
    setStatus("That todo is no longer available.", true);
    return;
  }

  if (action === "delete" && state.confirmDeleteTodoId !== todoId) {
    clearDeleteConfirmation();
    state.confirmDeleteTodoId = todoId;
    button.textContent = "Confirm Delete";
    state.confirmDeleteTimer = window.setTimeout(() => {
      if (state.confirmDeleteTodoId === todoId) {
        clearDeleteConfirmation();
        setStatus("Delete cancelled.");
      }
    }, 5000);
    setStatus("Click delete again within 5 seconds to confirm.");
    return;
  }

  if (action !== "delete") {
    clearDeleteConfirmation();
  }

  const previousTodos = cloneTodos(state.currentTodos);
  const previousCounts = { ...state.counts };

  try {
    if (action === "complete" || action === "reopen") {
      renderTodos(optimisticTodos(todoId, action));
      updateCounts(optimisticCounts(todo, action));
      setStatus(action === "complete" ? "Completing..." : "Reopening...");
      await fetchJson(`/api/todos/${encodeURIComponent(todoId)}/${action}`, {
        method: "POST",
        body: JSON.stringify({}),
      });
      setStatus(action === "complete" ? "Todo completed." : "Todo reopened.");
      void fullRefresh();
      return;
    }

    if (action === "calendar") {
      openCalendarModal(todo);
      return;
    }

    if (action === "delete") {
      clearDeleteConfirmation();
      renderTodos(optimisticTodos(todoId, action));
      updateCounts(optimisticCounts(todo, action));
      setStatus("Deleting...");
      await fetchJson(`/api/todos/${encodeURIComponent(todoId)}`, {
        method: "DELETE",
        body: JSON.stringify({}),
      });
      setStatus("Todo deleted.");
      void fullRefresh();
    }
  } catch (error) {
    clearDeleteConfirmation();
    renderTodos(previousTodos);
    updateCounts(previousCounts);
    setStatus(error.message, true);
  }
}

async function submitShare(event) {
  event.preventDefault();
  const submitButton = event.submitter || elements.shareSubmitButton;
  const todoId = elements.shareTodoId.value;
  const mode = elements.shareMode.value;
  const recipients = parseRecipients(elements.shareRecipients.value);

  if (!todoId || !mode) {
    return;
  }
  if (!recipients.length) {
    setStatus("Enter at least one email address.", true);
    return;
  }

  if (mode === "email") {
    const href = buildMailtoUrl(
      recipients,
      elements.shareSubject.value.trim(),
      elements.shareBody.value,
    );
    window.location.href = href;
    closeShareModal();
    setStatus("Opened your mail app.");
    return;
  }

  try {
    setButtonBusy(submitButton, true, "Sending...");
    await fetchJson(`/api/todos/${encodeURIComponent(todoId)}/invite`, {
      method: "POST",
      body: JSON.stringify({ attendees: recipients }),
    });
    closeShareModal();
    setStatus("Calendar invite sent.");
    void fullRefresh();
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    setButtonBusy(submitButton, false);
  }
}

async function submitCalendar(event) {
  event.preventDefault();
  const submitButton = event.submitter || elements.calendarSubmitButton;
  const todoId = elements.calendarTodoId.value;
  const selectedCalendars = Array.from(
    elements.calendarOptions.querySelectorAll('input[name="calendar_user"]:checked')
  ).map((input) => input.value);

  if (!todoId) {
    return;
  }
  if (!selectedCalendars.length) {
    setStatus("Select at least one calendar.", true);
    return;
  }

  try {
    setButtonBusy(submitButton, true, "Adding...");
    const payload = await fetchJson(`/api/todos/${encodeURIComponent(todoId)}/calendar`, {
      method: "POST",
      body: JSON.stringify({ calendar_users: selectedCalendars }),
    });
    closeCalendarModal();
    const created = Array.isArray(payload.calendar_events) ? payload.calendar_events : [];
    const createdUsers = created
      .map((item) => item && item.calendar_user)
      .filter(Boolean);
    const calendarText = createdUsers.length
      ? ` to ${createdUsers.join(", ")}`
      : ".";
    setStatus(`Added to Google Calendar${calendarText}`);
    void fullRefresh();
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    setButtonBusy(submitButton, false);
  }
}

function bindEvents() {
  elements.refreshButton.addEventListener("click", () => {
    fullRefresh({ message: "Refreshed." });
  });

  elements.addGroupToggleButton.addEventListener("click", () => toggleAddGroupPanel());
  elements.addGroupCancelButton.addEventListener("click", () => toggleAddGroupPanel(false));
  elements.addToggleButton.addEventListener("click", () => toggleAddPanel());
  elements.addCancelButton.addEventListener("click", () => toggleAddPanel(false));

  elements.createGroupForm.addEventListener("submit", createGroup);
  elements.createForm.addEventListener("submit", createTodo);
  elements.editForm.addEventListener("submit", submitEdit);
  elements.shareForm.addEventListener("submit", submitShare);
  elements.calendarForm.addEventListener("submit", submitCalendar);
  elements.moveForm.addEventListener("submit", submitMove);
  elements.advanceForm.addEventListener("submit", submitAdvance);
  elements.closeModalButton.addEventListener("click", closeEditModal);
  elements.closeShareModalButton.addEventListener("click", closeShareModal);
  elements.closeCalendarModalButton.addEventListener("click", closeCalendarModal);
  elements.closeMoveModalButton.addEventListener("click", closeMoveModal);
  elements.closeAdvanceModalButton.addEventListener("click", closeAdvanceModal);
  elements.closeChainModalButton.addEventListener("click", closeChainModal);
  elements.shareCancelButton.addEventListener("click", closeShareModal);
  elements.calendarCancelButton.addEventListener("click", closeCalendarModal);
  elements.moveCancelButton.addEventListener("click", closeMoveModal);
  elements.advanceCancelButton.addEventListener("click", closeAdvanceModal);
  elements.clearDueButton.addEventListener("click", () => {
    elements.editDueAt.value = "";
  });
  elements.modalBackdrop.addEventListener("click", (event) => {
    if (event.target === elements.modalBackdrop) {
      closeEditModal();
    }
  });
  elements.shareModalBackdrop.addEventListener("click", (event) => {
    if (event.target === elements.shareModalBackdrop) {
      closeShareModal();
    }
  });
  elements.calendarModalBackdrop.addEventListener("click", (event) => {
    if (event.target === elements.calendarModalBackdrop) {
      closeCalendarModal();
    }
  });
  elements.moveModalBackdrop.addEventListener("click", (event) => {
    if (event.target === elements.moveModalBackdrop) {
      closeMoveModal();
    }
  });
  elements.advanceModalBackdrop.addEventListener("click", (event) => {
    if (event.target === elements.advanceModalBackdrop) {
      closeAdvanceModal();
    }
  });
  elements.chainModalBackdrop.addEventListener("click", (event) => {
    if (event.target === elements.chainModalBackdrop) {
      closeChainModal();
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
