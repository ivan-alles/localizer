:: Deploy on GitHub pages.
:: See https://cli.vuejs.org/guide/deployment.html#github-pages

call yarn build

pushd dist

git init
git add -A
git commit -m 'deploy'

git push -f git@github.com:ivan-alles/localizer.git master:gh-pages

popd
