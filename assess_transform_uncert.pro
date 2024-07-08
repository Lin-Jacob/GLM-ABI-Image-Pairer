; asssess the transformation uncertainty with the matched keypoints
; test with the polywarp and poly_2D transformation
;
pro assess_transform_uncert, imagesave=imagesave, xdist=xdist, ydist=ydist, dist=dist, ipa=ipa

!path = '~fangfang.yu/tools:'+!path

; ... read in the CNN matched keypoints ....
dir_data = '/data/home004/fangfang.yu/Moon/AI_CNN/'

list_phaseangle = [-65, -43, -30, -6,  16, 25, 60, 70]
str_phaseangle = strcompress(string(list_phaseangle), /remove_all)

if n_elements(ipa) eq 0 then ipa=2


; ... read in the CNN matched keypoints ....
dir_data = '/data/data461/fyu/Moon/AI_CNN/'

; ... angle to file ...
fname = dir_data + 'angle_to_file.txt'
nrec = file_lines(fname)-1
str = ' '
phaseangle = intarr(nrec)
sfnames = strarr(nrec)
stimetag = strarr(nrec)

openr, rlun, fname, /get_lun
for irec=0, nrec-1 do begin
  readf, rlun, str
  res = strsplit(str, ':', /extract)
  phaseangle[irec] = fix(res[0])
  sfnames[irec] = strcompress(res[1], /remove_all)
  
  token = strsplit(res[1], '_', /extract, count=nstr)
  stimetag[irec] = token[nstr-2]
  
endfor
readf, rlun, str
free_lun, rlun


; ... keypoints files, named with the source time tags...
spawn, 'ls '+dir_data+'/updated/2*.txt', tfnames, count=nfile

ttimetag = strarr(nrec)
for ifile=0, nfile-1 do begin
  res = strsplit(tfnames[ifile], '/', /extract, count=nstr)
  token = strsplit(res[nstr-1], '.', /extract)
  ttimetag[ifile] = token[0]
endfor

; ... find the keypoint files for the given phase angle ...
index = where(stimetag[ipa] eq ttimetag)
if index[0] eq -1 then stop

fname = tfnames[index[0]]

print, fname

nrec = file_lines(fname)

xin = intarr(2, nrec)
xpin = intarr(2, nrec)
x1 = intarr(nrec)
y1 = intarr(nrec)
x2 = intarr(nrec)
y2 = intarr(nrec)

str = ' '
openr, rlun, fname, /get_lun
for irec=0, nrec-1 do begin
  readf, rlun, str
  res = strsplit(str, '-', /extract, count=nstr)
  if nstr ne 2 then stop
  
  tk1 = strsplit(res[0], ',', /extract, count=ntk1)
  if ntk1 ne 2 then stop
  xin[0, irec]=  fix(tk1[0])
  xin[1, irec] = fix(tk1[1])
  
  x1[irec] = fix(tk1[0])
  y1[irec] = fix(tk1[1])
  
  
  tk2 = strsplit(res[1], ',', /extract, count=ntk2)
  if ntk2 ne 2 then stop
  xpin[0, irec]=  fix(tk2[0])
  xpin[1, irec] = fix(tk2[1])  
  
  x2[irec] = fix(tk2[0])
  y2[irec] = fix(tk2[1])
  
endfor
free_lun, rlun


; >>> transform <<<
fitorder= 2
POLYWARP, x2, y2, x1, y1, fitorder, Kx, Ky

; ... ABI image size ...
n_line = 1400
n_elem = 1600

; ... validation of transfomation ...
A = fltarr(n_elem, n_line)
x1p = intarr(nrec)
y1p = intarr(nrec)

for irec =0, nrec-1 do begin
  A[*, *] = 0.
  A[x2[irec], y2[irec]] = 250.
  
  B = poly_2D(A, Kx, Ky)
  
  indx = where(B eq max(B))
  x1p[irec] = indx[0] mod n_elem
  y1p[irec] = indx[0]/n_elem
  
endfor

; ... difference ...
x_diff = float(x1p - x1)
y_diff = float(y1p - y1)

print, max(x_diff), min(x_diff), mean(x_diff), median(x_diff), stddev(x_diff)
print, max(y_diff), min(y_diff), mean(y_diff), median(y_diff), stddev(y_diff)


; ... plot the histogram ...

; x difference 
binsize = 1
nrec = n_elements(x_diff)

x_freq = histogram(x_diff, binsize=binsize, locations=x_loc)*100./float(nrec)
y_freq = histogram(y_diff, binsize=binsize, locations=y_loc)*100./float(nrec)

d_dist = sqrt(x_diff^2 + y_diff^2)
d_freq = histogram(d_dist, binsize=binsize, locations=d_loc)*100./float(nrec)

threshold = 3		; mismatched distance < 5 pixels

if keyword_set(xdist) then begin
  freq = x_freq
  xx = x_loc
  title = 'Histogram  (x direction) for PA at ' + str_phaseangle[ipa] +' Degrees'
  ind = where(abs(x_diff) le threshold)
  nrec5 = n_elements(ind)
  strline = string(nrec-nrec5, nrec5, mean(x_diff[ind]), stddev(x_diff[ind]), $
    format='("(> 3 pixels): #", i3, ",  (<= 3 pixels): #", i4, ",  mean=", f4.1, ",  stdv=", f4.1)')
    
  fout = 'transform_mismatch_hist_pa'+str_phaseangle[ipa]+'_xdist.png'
endif
if keyword_set(ydist) then begin
  freq = y_freq
  xx = y_loc
  title = 'Histogram (y direction) for PA at ' + str_phaseangle[ipa] +' Degrees'
  ind = where(abs(y_diff) le threshold)
  nrec5 = n_elements(ind)  
  strline = string(nrec-nrec5, nrec5, mean(y_diff[ind]), stddev(y_diff[ind]), $
    format='("(> 3 pixels): #", i3, ",  (<= 3 pixels): #", i4, ",  mean=",f4.1, ",  stdv=",f5.2)')
    
  fout = 'transform_mismatch_hist_pa'+str_phaseangle[ipa]+'_ydist.png'  
endif
if keyword_set(dist) then begin
  freq = d_freq
  xx = d_loc
  title = 'Histogram for PA at ' + str_phaseangle[ipa] +' Degrees'
  ind = where(abs(d_dist) le threshold)
  nrec5 = n_elements(ind)
  strline = string(nrec-nrec5, nrec5, mean(d_dist[ind]), stddev(d_dist[ind]), $
    format='("(> 3 pixels): #", i3, ",  (<= 3 pixels): #", i4, ",  mean=",f4.1, ",  stdv=",f4.1)')
    
  fout = 'transform_mismatch_hist_pa'+str_phaseangle[ipa]+'_dist.png'
endif

ytitle = 'Frequency (%)'
xtitle = 'Mismatched Distance (in Pixel)'

xrange = [-10., 10.]
if keyword_set(dist) then xrange=[-1, 10]

y_hi = max(freq)
yrange = [0, 1.2*y_hi]

delta_x = xrange[1] - xrange[0]
delta_y = yrange[1] - yrange[0]

xsize = 700 & ysize=450
Fsize = 12
 
p = plot (xrange, yrange, /nodata, dim=[xsize, ysize], $
  FONT_COLOR='Black', Font_size=Fsize1, $
  thick=2., $
  xthick=2., ythick=2., $
  ystyle=1, $ 
  xstyle=1, $
  title = title, $
  yrange=yrange, xrange=xrange, $
  ytitle = ytitle, $
  xtitle = xtitle, $
  pos=[0.12, 0.12, 0.95, 0.90])

;p = plot(xx, freq, linestyle=0, thick=2, 'Blue', /overplot)

index = where(freq gt 0.)
p = plot(xx[index], freq[index], symbol='o', sym_color='Blue', /sym_filled, /overplot)

t=text([xrange[0]+delta_x*0.05], [yrange[1]-0.1*delta_y], strline, /data, color='black', font_size=Fsize-4)

if keyword_set(imagesave) then p.save, fout, resolution=100
print, 'save to = ', fout

stop


END
