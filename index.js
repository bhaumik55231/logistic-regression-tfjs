const tf = require('@tensorflow/tfjs');

const w0 = tf.scalar(Math.random()).variable();
const w1 = tf.scalar(Math.random()).variable();

const xs = tf.tensor1d([5.1, 4.7, 6.4, 7.0]);
const ys = tf.tensor1d([1, 1, 0, 0]);

const f = x => tf.div(1, tf.add(1, ((w0.add(w1.mul(x))).neg()).exp()))

const loss = (pred, label) => ((pred.sub(label)).square()).mean();

const optimizer = tf.train.sgd(0.01);

// Train the model.
for (let i = 0; i < 100; i++) {
  optimizer.minimize(() => loss(f(xs), ys));
}

// Make predictions.
const preds = f(xs).dataSync();
preds.forEach((pred, i) => {
   console.log(`x: ${i}, pred: ${pred}`);
});