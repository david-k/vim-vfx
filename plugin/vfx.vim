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

	nnoremap <buffer> <silent> <CR> :py3 vfx.open_entry()<CR>
	nnoremap <buffer> <silent> <C-CR> :py3 vfx.open_entry(split_window=True)<CR>
	nnoremap <buffer> <silent> <Backspace> :py3 vfx.move_up()<CR>
	nnoremap <buffer> <silent> _ :py3 vfx.open_vim_cwd()<CR>
	nnoremap <buffer> <silent> ´ :py3 vfx.change_vim_cwd()<CR>
	nnoremap <buffer> <silent> g. :py3 vfx.toggle_dotfiles()<CR>
	nnoremap <buffer> <silent> gl :py3 vfx.toggle_details()<CR>
	nnoremap <buffer> <silent> <Tab> :py3 vfx.toggle_expand()<CR>
	nnoremap <buffer> <silent> - :py3 vfx.quit()<CR>
	nnoremap <buffer> <silent> ^ :py3 vfx.move_cursor_to_filename()<CR>
	nnoremap <buffer> <silent> yp :py3 vfx.yank_absolute_path()<CR>
	nnoremap <buffer> <silent> yP :py3 vfx.yank_absolute_path(use_system_clipboard=True)<CR>


	autocmd BufUnload <buffer> py3 vfx.on_buf_unload()
	autocmd BufWriteCmd <buffer> py3 vfx.on_buf_save()
	autocmd InsertLeave <buffer> py3 vfx.buffer_changed()
	autocmd TextChanged <buffer> py3 vfx.buffer_changed()
	autocmd DirChanged window py3 vfx.change_vim_cwd()

	" TODO Define `autocmd FileExplorer`. This makes it possible to do `:edit <DIRECTORY>` like netrw

	setl conceallevel=2
	setl concealcursor=nvic
	setl textwidth=0
	setl wrapmargin=0
	setl indentexpr=py3eval('vfx.get_indent_for_vim('.v:lnum.')')
	setl cursorline
endfunction
