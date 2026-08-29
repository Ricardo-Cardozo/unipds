const gulp = require('gulp');
const connect = require('gulp-connect');
const audiosprite = require('./vendor/audiosprite');
const glob = require('glob');
const shell = require('gulp-shell');
const fs = require('fs');

gulp.task('audio', gulp.parallel(function(cb) {
  const files = glob.sync('./src/assets/sounds/*.mp3');
  const outputPath = './dist/audio';
  const opts = {
    output: outputPath,
    path: './',
    format: 'howler2',
    'export': 'ogg,mp3',
    loop: ['quacking', 'sniff']
  };

  return audiosprite(files, opts, function(err, obj) {
    if (err) {
      // O erro precisa aparecer no terminal para orientar quem executa a tarefa da aula.
      // eslint-disable-next-line no-console
      console.error(err);
    }

    return fs.writeFile('./dist/audio' + '.json', JSON.stringify(obj, null, 2), cb);
  });
}));

gulp.task('images', gulp.parallel(function(){
  // There is a texturepacker template for spritesmith but it doesn't work
  // well with complex directory structures, so instead we use the shell
  // checking TexturePacker --version first ensures it bails if TexturePacker
  // isn't installed
  return gulp.src('*', {read:false})
    .pipe(shell([
      'TexturePacker --version || echo ERROR: TexturePacker not found, install TexturePacker',
      'TexturePacker --disable-rotation --data dist/sprites.json --format json ' +
        '--sheet dist/sprites.png src/assets/images'
    ]))
    .pipe(connect.reload());
}));

gulp.task('deploy', gulp.parallel(function() {
  return gulp.src('*', {read:false})
    .pipe(shell([
      'aws --profile duckhunt s3 sync dist/ s3://duckhuntjs.com --include \'*\' --acl \'public-read\''
    ]));
}));

gulp.task('default', gulp.parallel('images', 'audio'));
