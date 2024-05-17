// script.js

// Import TensorFlow.js and tfjs-vis
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

// Define a simple GAT layer
class GraphAttentionLayer extends tf.layers.Layer {
  constructor(config) {
    super(config);
    this.units = config.units;
  }

  build(inputShape) {
    this.kernel = this.addWeight('kernel', [inputShape[1], this.units], 'float32', tf.initializers.glorotUniform());
    this.built = true;
  }

  call(inputs, kwargs) {
    const x = inputs[0];
    const adjacency = inputs[1];
    const h = tf.matMul(x, this.kernel);
    const attention = tf.matMul(adjacency, h);
    return tf.matMul(attention, this.kernel, false, true);
  }

  computeOutputShape(inputShape) {
    return [inputShape[0], this.units];
  }
}

tf.serialization.registerClass(GraphAttentionLayer);

// Create the model
const createModel = (inputShape) => {
  const inputFeatures = tf.input({shape: inputShape});
  const adjacencyMatrix = tf.input({shape: [inputShape[0]]});

  const gatLayer = new GraphAttentionLayer({units: 8});
  const output = gatLayer.apply([inputFeatures, adjacencyMatrix]);

  return tf.model({inputs: [inputFeatures, adjacencyMatrix], outputs: output});
};

// Create dummy data for a simple graph
const features = tf.tensor2d([[1, 0], [0, 1], [1, 1]], [3, 2]);
const adjacency = tf.tensor2d([[1, 1, 0], [1, 1, 1], [0, 1, 1]], [3, 3]);

// Initialize the model
const model = createModel([2]);

// Display the model structure using tfjs-vis
tfvis.show.modelSummary({name: 'GAT Model Summary', tab: 'Model'}, model);

// Display the input data
const container = {name: 'Input Features', tab: 'Data'};
tfvis.render.table(container, {headers: ['Feature 1', 'Feature 2'], values: features.arraySync()});

const adjacencyContainer = {name: 'Adjacency Matrix', tab: 'Data'};
tfvis.render.table(adjacencyContainer, {headers: ['Node 1', 'Node 2', 'Node 3'], values: adjacency.arraySync()});

// Train the model and visualize training progress
const train = async () => {
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError
  });

  const history = await model.fit([features, adjacency], features, {
    epochs: 20,
    callbacks: tfvis.show.fitCallbacks(
      {name: 'Training Performance', tab: 'Training'},
      ['loss'],
      {callbacks: ['onEpochEnd']}
    )
  });

  console.log('Training complete');
};

train();
