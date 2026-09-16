const collisionMode = (value) => {
  if (value === true) return 'rigid';
  if (value === false || value == null) return 'none';
  return String(value);
};

class VfLiveWorldReference {
  #layers = [];
  #physics = [];

  constructor(specification) {
    if (!specification?.id) throw new Error('A live world requires an id.');
    this.id = String(specification.id);
    this.label = String(specification.label ?? specification.id);
  }

  // add() layers data into this world. A time axis is optional: pushed physics
  // advances every layer, including point clouds that only contain x_i/y_i.
  add(data) {
    if (!data || typeof data !== 'object') throw new Error('world.add(data) requires an object.');
    this.#layers.push(data);
    return this;
  }

  // push() attaches an advancing local rule. Collision policy belongs to the
  // rule, so gravity can use rigid contacts or intentionally pass through.
  push(operator) {
    if (!operator || typeof operator.advance !== 'function') {
      throw new Error('world.push(physics) requires an advance function.');
    }
    this.#physics.push({ ...operator,
      id: String(operator.id ?? `physics-${this.#physics.length + 1}`),
      collisions: collisionMode(operator.collisions),
    });
    return this;
  }

  read(key) {
    for (let index = this.#layers.length - 1; index >= 0; index -= 1) {
      if (Object.prototype.hasOwnProperty.call(this.#layers[index], key)) {
        return this.#layers[index][key];
      }
    }
    return undefined;
  }

  advance(context = {}) {
    for (const operator of this.#physics) operator.advance({ world: this, ...context });
  }

  reset(context = {}) {
    for (const operator of this.#physics) operator.reset?.({ world: this, ...context });
  }

  snapshot() {
    return Object.freeze({ id: this.id, label: this.label,
      layerCount: this.#layers.length,
      physics: this.#physics.map(({ id, collisions }) => ({ id, collisions })),
    });
  }
}

export function createVfLiveWorldStackReference() {
  const worlds = [];
  let activeIndex = -1;
  return Object.freeze({
    append(specification) {
      const world = new VfLiveWorldReference(specification);
      if (worlds.some((candidate) => candidate.id === world.id)) {
        throw new Error(`Live world '${world.id}' already exists.`);
      }
      worlds.push(world);
      if (activeIndex < 0) activeIndex = 0;
      return world;
    },
    flip(id) {
      const index = worlds.findIndex((world) => world.id === id);
      if (index < 0) throw new Error(`Unknown live world '${id}'.`);
      activeIndex = index;
      return worlds[activeIndex];
    },
    get active() { return worlds[activeIndex] ?? null; },
    snapshots() { return worlds.map((world) => world.snapshot()); },
  });
}
