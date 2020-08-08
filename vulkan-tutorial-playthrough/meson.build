project('Vulkan Tutorial Playthrough', 'cpp',  default_options : ['cpp_std=c++2a'])

deps = [
    dependency('glfw3'),
    dependency('vulkan')
]

executable('app', [
    'source/main.cxx'
], dependencies : deps)



source = join_paths(meson.source_root(), 'shader.frag')
dest = join_paths(meson.build_root(), 'frag.spv')
r = run_command('glslc', source, '-o', dest)

source = join_paths(meson.source_root(), 'shader.vert')
dest = join_paths(meson.build_root(), 'vert.spv')
r = run_command('glslc', source, '-o', dest)