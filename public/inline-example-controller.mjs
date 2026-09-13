function displayValue(execution) {
  if (typeof execution.output?.stdout === "string") return execution.output.stdout;
  if (execution.packets) return "Program emitted UI output.";
  if (typeof execution.output === "string") return execution.output;
  if (execution.output === undefined) return "Program completed.";
  if (execution.output?.kind === "console" && Array.isArray(execution.output.values)) {
    return execution.output.values.map((value) => JSON.stringify(value)).join("\n");
  }
  return JSON.stringify(execution.output, null, 2);
}

export function createInlineExampleController({ runner, view }) {
  let generation = 0;
  return Object.freeze({
    cancel() {
      generation += 1;
    },
    async run(source) {
      const currentGeneration = ++generation;
      view.start();
      try {
        const execution = await runner.run(source);
        if (currentGeneration !== generation) return;
        view.showTerminal(displayValue(execution));
        view.hideResult();
        if (execution.packets) view.showResult(execution.packets, execution.timing);
      } catch (error) {
        if (currentGeneration !== generation) return;
        view.showTerminal(`${error.message}. No fallback result was rendered.`);
        view.hideResult();
      } finally {
        if (currentGeneration === generation) view.finish();
      }
    },
  });
}
