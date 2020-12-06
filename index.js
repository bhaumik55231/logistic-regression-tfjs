const tf = require('@tensorflow/tfjs');

const w0 = tf.scalar(Math.random()).variable();
const w1 = tf.scalar(Math.random()).variable();
const ones = tf.ones([1,1]);

const xs = tf.tensor1d([5.1, 4.7, 6.4, 7.0]);
const ys = tf.tensor1d([1, 1, 0, 0]);

// xi=>1/(1+Math.exp(-(P[0]+(P[1]*xi))))
const f = x => ones.div(ones.sum(((w0.sum(w1.mul(x))).neg).exp))

const loss = (pred, label) => tf.mean(tf.square(tf.sub(pred, label)));

console.log(f(xs))
const optimizer = tf.train.sgd(0.01);

// Train the model.
for (let i = 0; i < 10; i++) {
   optimizer.minimize(() => loss(f(xs), ys));
}

// Make predictions.
const preds = f(xs).dataSync();
preds.forEach((pred, i) => {
   console.log(`x: ${i}, pred: ${pred}`);
});