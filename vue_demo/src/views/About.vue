<!-- Copyright 2016-2020 Ivan Alles. See also the LICENSE file. -->

<template>
  <div class="content">
    <header>
      <h1>About</h1>
      <p>
      This <a href="https://github.com/ivan-alles/preference-model">app</a> demonstrates how a neural network learns what you like.
      </p>
    </header>
    <section>
      <h2>How it works</h2>
      <p>
      In the beginning, you will see random pictures, dreamed up by a neural network. This neural network is called a generator. 
      The people on the images look real, but they do not exist!
      </p>
      <p>
      When you like some pictures, the AI will learn them. The generator starts to create new images 
      sharing the common traits of your likes. 
      </p>
      <p>
      You can vary the pictures less or more by changing the <b>Variance</b>.
      </p>
      <p>
      Now you can like the best of the new pictures to create even more beautiful ones. Eventually, you can let the AI create a person of your dream!
      </p>
    </section>
    <section>
      <h2>Credits</h2>
      <p>
      The generator is from the <a href="https://github.com/tkarras/progressive_growing_of_gans">Progressive Growing of GANs for Improved Quality, Stability, and Variation</a> by Tero Karras (NVIDIA), Timo Aila (NVIDIA), Samuli Laine (NVIDIA), Jaakko Lehtinen (NVIDIA and Aalto University).
      </p>
    </section>
  </div>
</template>

<style>
.content {
  max-width: 960px;
  margin: 10px;
}
</style>
