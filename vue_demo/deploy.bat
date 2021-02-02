:: Deploy on GitHub pages.
:: See https://cli.vuejs.org/guide/deployment.html#github-pages

call yarn build

pushd dist

:: This fixes the problem of Github Pages showing ERROR 404 for SPA routes.
copy index.html about.html

git init
git add -A
git commit -m 'deploy'

git push -f git@github.com:ivan-alles/preference-model.git master:gh-pages

popd
