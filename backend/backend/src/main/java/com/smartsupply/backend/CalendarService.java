package com.smartsupply.backend;

import org.springframework.stereotype.Service;
import java.io.BufferedReader;
import java.io.FileReader;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

@Service
public class CalendarService {

    public List<CalendarDTO> readCalendarCsv() {
        List<CalendarDTO> records = new ArrayList<>();
        
        Path csvPath = Paths.get(System.getProperty("user.dir"), "data", "raw", "dim_calendar_vietnam_2026.csv");
        
        try (BufferedReader br = new BufferedReader(new FileReader(csvPath.toFile()))) {
            String line;
            boolean isHeader = true;
            
            while ((line = br.readLine()) != null) {
                // 1. Bỏ qua dòng tiêu đề đầu tiên (calendar_date, day_of_week...)
                if (isHeader) {
                    isHeader = false;
                    continue;
                }
                
                // 2. Cắt dòng thành mảng các chữ dựa theo dấu phẩy
                String[] values = line.split(",", -1);
                
                // 3. Nếu đủ 5 cột thì tiến hành "đổ" vào khuôn CalendarDTO
                if (values.length >= 5) {
                    CalendarDTO dto = new CalendarDTO();
                    dto.setCalendarDate(values[0].trim());
                    dto.setDayOfWeek(Integer.parseInt(values[1].trim()));
                    dto.setIsHoliday(Integer.parseInt(values[2].trim()));
                    dto.setEventType(values[3].trim());
                    dto.setIsPayloadDay(Integer.parseInt(values[4].trim()));
                    
                    // Thêm đối tượng đã sạch sẽ vào danh sách
                    records.add(dto);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        
        return records;
    }
}