import "@testing-library/jest-dom/vitest";

// jsdom has no ResizeObserver (cmdk measures its list with one); browsers do. Test-only stub.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver ??= ResizeObserverStub as unknown as typeof ResizeObserver;
// jsdom also lacks scrollIntoView, which cmdk calls on the selected item.
Element.prototype.scrollIntoView ??= function scrollIntoView() {};
