package com.example.demo.controller;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.util.FileCopyUtils;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.*;

@RestController
@RequestMapping("/api")
public class ImageRestoreController {

    @Value("${restore.temp-dir:./temp}")
    private String tempDir;

    @PostMapping("/restore-image")
    public ResponseEntity<byte[]> restoreImage(@RequestParam("file") MultipartFile file)
            throws IOException, InterruptedException {
        // 1. 处理临时目录为绝对路径
        String tempDirPath = tempDir.trim();
        File dir = new File(tempDirPath);
        if (!dir.isAbsolute()) {
            dir = new File(System.getProperty("user.dir"), tempDirPath);
        }
        if (!dir.exists())
            dir.mkdirs();
        String inputFileName = System.currentTimeMillis() + "_" + file.getOriginalFilename();
        File inputFile = new File(dir, inputFileName);
        // 确保父目录存在
        File parent = inputFile.getParentFile();
        if (!parent.exists()) {
            parent.mkdirs();
        }
        file.transferTo(inputFile);

        // 2. 调用Python模型（假设模型输出到 output.png）
        String outputFileName = "output_" + System.currentTimeMillis() + ".png";
        File outputFile = new File(dir, outputFileName);
        ProcessBuilder pb = new ProcessBuilder("python", "your_model_script.py", inputFile.getAbsolutePath(),
                outputFile.getAbsolutePath());
        pb.inheritIO();
        Process process = pb.start();
        process.waitFor();

        // 3. 读取恢复后的图片
        byte[] imageBytes = FileCopyUtils.copyToByteArray(outputFile);

        // 4. 返回图片流
        return ResponseEntity.ok()
                .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=restored.png")
                .contentType(MediaType.IMAGE_PNG)
                .body(imageBytes);
    }
}