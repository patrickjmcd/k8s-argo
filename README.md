# k8s-argo


## Helpers

### Resizing a PVE HD

#### 1st step:  increase/resize disk  from  GUI console

From PVE GUI, select the VM -> Hardware -> Select the Disk -> Resize disk

#### 2nd step : Extend physical drive partition

1. check free space: 
```shell
sudo fdisk -l
```
2. Exend physical partition:
```shell
sudo growpart /dev/sda 3
```

3. Check physical drive:
```shell
sudo pvdisplay
```

4. Instruct LVM that disk size has changed:
```shell
sudo pvresize /dev/sda3
```

5. Check physical drive if has changed:
```shell
sudo pvdisplay
```

#### 3rd step:  Extend  Logical  volume

1. Check Logical Volume
```shell
sudo lvdisplay
```

2. Extend Logical Volume
```shell
sudo lvextend -l +100%FREE /dev/ubuntu-vg/ubuntu-lv
```

3. Check Logical Volume if has changed
```shell
sudo lvdisplay
```

#### 4th  step :   Resize Filesystem

1. Resize filesystem
```shell
sudo resize2fs /dev/ubuntu-vg/ubuntu-lv
```
2. Confirm results
```shell
sudo fdisk -l
```
