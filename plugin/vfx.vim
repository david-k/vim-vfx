if exists('g:loaded_vfx')
  finish
endif
let g:loaded_vfx = 1


" Main functions
"-------------------------------------------------------------------------------
function! Vfx() abort
py3 << EOF
import vfx
vfx.init()
EOF

	nnoremap <buffer> <CR> :py3 vfx.open_entry()<CR>
	nnoremap <buffer> <Backspace> :py3 vfx.move_up()<CR>
	nnoremap <buffer> g. :py3 vfx.toggle_dotfiles()<CR>
	nnoremap <buffer> <Tab> :py3 vfx.toggle_expand()<CR>
	nnoremap <buffer> - :py3 vfx.quit()<CR>


	autocmd BufUnload <buffer> py3 vfx.on_buf_unload()
endfunction


" Syntax highlighting
"-------------------------------------------------------------------------------
autocmd FileType vfx syntax match Directory ".*/$"
