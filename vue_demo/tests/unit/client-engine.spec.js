// Copyright 2016-2020 Ivan Alles. See also the LICENSE file. 

import * as tf from '@tensorflow/tfjs';
tf.setBackend('cpu')

import { testables } from '@/client-engine'
const PreferenceModel = testables.PreferenceModel;
const scaledDirichlet = testables.scaledDirichlet;
const cartesianToSpherical = testables.cartesianToSpherical;
const sphericalToCartesian = testables.sphericalToCartesian;


test('tensorflow installed correctly', () => {
  const t = tf.tensor1d([1, 2, 3]);
  expect(t.shape).toStrictEqual([3]);
});

const SHAPE=512;

describe('PreferenceModel', () => {
  const model = new PreferenceModel(SHAPE);
  model.init();

  test('initially generate random values', () => {
    expect(model.isRandom).toStrictEqual(true);
  });

  test('Shape and size of random values', () => {
    let x = model.generate(1, 0);
    expect(x.shape).toStrictEqual([1, SHAPE]);

    x = model.generate(3, 0);
    expect(x.shape).toStrictEqual([3, SHAPE]);
  });

  test('Train on one sample', () => {
    model.train(tf.randomNormal([1, SHAPE]));
    let x = model.generate(1, 0);
    expect(x.shape).toStrictEqual([1, SHAPE]);
    x = model.generate(3, 0);
    expect(x.shape).toStrictEqual([3, SHAPE]);
  });

  test('Reset training', () => {
    model.train(tf.tensor([]));
    let x = model.generate(1, 0);
    expect(x.shape).toStrictEqual([1, SHAPE]);
  });

});

describe.each([
  [5000, 2, 1, 1],
  [5000, 2, 3, 1],
  [5000, 2, .5, 1],
  [5000, 2, .5, 0.1],
  [5000, 2, 1.5, 10],
  [5000, 3, 4, 1.5],
  [5000, 3, .4, 0.7],
  [10000, 5, .4, 0.7],
  [10000, 5, 4, 0.8],
])('scaledDirichlet(%i, %f, %f, %f)', (n, k, a, scale) => {
  const x = scaledDirichlet([n], k, a, scale);

  test('output is in expected range', () => {
    expect(x.shape).toStrictEqual([n, k]);
    expectTensorsClose(x.sum(1), tf.ones([n]));
    expectTensorsEqual(tf.greaterEqual(x, (1 - scale) / k), tf.ones([n, k]));
    expectTensorsEqual(tf.lessEqual(x, (scale * (k-1) + 1) / k), tf.ones([n, k]));
  });
  test('output has correct mean and std ', () => {
    expectTensorsClose(x.mean(0), tf.fill([k], 1 / k), 0.1);
    const stdExp = scale * Math.sqrt((k-1) / k**2 / (k * a + 1));
    expectTensorsClose(std(x, 0), tf.fill([k], stdExp), 0.1)
  });
});


describe.each([
  [[[0.1]]],
  [[[-.5]]],
  [[[0.2], [0.1], [-0.3]]],
])('sphericalToCartesian(%o) for 2d vectors', (phi) => {
  function getExpected(phi) {
    return tf.tensor(phi.map(x => [Math.cos(x[0]), Math.sin(x[0])]));
  }

  const x = sphericalToCartesian(tf.tensor(phi));

  test('output has expected value', () => {
    expectTensorsClose(x, getExpected(phi), 0.0001);
  });
});

describe.each([
  [[[0.1, 0.2]]],
  [[[-0.1, 0.3]]],
  [[[0.2, -0.3], [-0.01, 0], [-0.7, 0]]],
])('sphericalToCartesian(%o) for 3d vectors', (phi) => {
  function getExpected(phi) {
    return tf.tensor(phi.map(x => [
      Math.cos(x[0]), 
      Math.sin(x[0]) * Math.cos(x[1]),
      Math.sin(x[0]) * Math.sin(x[1])
    ]));
  }

  const x = sphericalToCartesian(tf.tensor(phi));

  test('output has expected value', () => {
    expectTensorsClose(x, getExpected(phi), 0.0001);
  });
});

describe.each([
  [[[0.5, 1]]],
  [[[-0.1, 0.3]]],
  [[[0, 0.3]]],
  [[[0, -0.3]]],
  [[[-0.01, 0]]],
  [[[2, 0]]],
  [[[-2, 0]]],
  [[[0.2, -0.3], [-0.01, 0], [0, 11]]],
  [[[0.5, 1, 2]]],
  [[[0.5, 1, -2]]],
  [[[0.5, 1, 0]]],
  [[[1, 3, 1]]],
  [[[-1, -3, 1]]],
  [[[-1, 1, 0]]],
  [[[-1, -1, 0]]],
  [[[ 1, 0, 0]]],
  [[[-1, 0, 0]]],
  [[[-2, 0, 1], [0, 0, 1]]],
])('cartesianToSpherical(%o)', (x) => {

  const phi = cartesianToSpherical(tf.tensor(x));

  test('output has expected value', () => {
    const phiExp = tf.tensor(x.map((e) => cartesianToSphericalSimple(e)));
    expectTensorsClose(phi, phiExp, 0.0001);
  });
});

describe('Cartesian and spherical back and forth', () => {

  for(let t = 0; t < 100; ++t) {
    const n = Math.floor(Math.random() * 4 + 2);
    const size = Math.floor(Math.random() * 10 + 1);
    let x = tf.randomUniform([size, n], -10, 10);
    x = x.div(tf.sqrt(x.square().sum(1, true)));

    const phi = cartesianToSpherical(x);
    test('phi has correct shape and range', () => {
      expect(phi.shape).toStrictEqual([size, n - 1]);

      const phiParts = phi.split([n-2, 1], 1);

      expectTensorsEqual(phiParts[0].greaterEqual(0), tf.onesLike(phiParts[0]));
      expectTensorsEqual(phiParts[0].lessEqual(Math.PI), tf.onesLike(phiParts[0]));

      expectTensorsEqual(phiParts[1].greaterEqual(-Math.PI), tf.onesLike(phiParts[1]));
      expectTensorsEqual(phiParts[1].lessEqual(Math.PI), tf.onesLike(phiParts[1]));
    });

    test('converted back x1 is close to the original x', () => {
      const x1 = sphericalToCartesian(phi);
      expectTensorsClose(x1, x, 0.001);
    });
  }

});

/**
 * Check if tensors are close. Is needed as tf.test_util.expectArraysClose()
 * always succeedes with tensor arguments.
 *
 */
function expectTensorsClose(actual, expected, epsilon=null) {
  tf.test_util.expectArraysClose(
    actual.arraySync(),
    expected.arraySync(),
    epsilon);
}

/**
 * Check if tensors are close. Is needed as tf.test_util.expectArraysClose()
 * always succeedes with tensor arguments, even if they are not equal (probably due to a bug).
 */
function expectTensorsEqual(actual, expected) {
  tf.test_util.expectArraysEqual(
    actual.arraySync(),
    expected.arraySync());
}

function std(x, axis=null) {
  const n = x.shape[axis];
  const m = tf.mean(x, axis)
  const xc = tf.sub(x, m)
  const x2 = tf.square(xc)
  return tf.sqrt(tf.div(tf.sum(x2, axis), n));
}

function norm(x) {
  return Math.sqrt(x.reduce((s, v) => s + v*v, 0));
}

function cartesianToSphericalSimple(x) {
    const n = x.length;
    const phi = Array(n - 1).fill(0);
    let k = -1;
    for (let i = n - 1; i > -1; --i) {
        if(x[i] != 0) {
            k = i;
            break;
        }
    }
    if(k == -1) {
      return phi;
    }

    for(let i = 0; i < k; ++i) {
        phi[i] = Math.acos(x[i] / norm(x.slice(i)));
    }

    if (k == n - 1) {
        phi[n - 2] = Math.acos(x[n-2] / norm(x.slice(-2)));
        if (x[n-1] < 0) {
            phi[n - 2] *= -1;
        }
    }
    else {
        phi[k] = x[k] > 0 ? 0 : Math.PI;
    }

    return phi;
}